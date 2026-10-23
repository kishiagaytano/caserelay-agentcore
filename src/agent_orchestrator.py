"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline (uses the AgentCore CLI)
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)

Deployment uses the AgentCore CLI (`agentcore`, npm package @aws/agentcore,
https://github.com/aws/agentcore-cli) through the pre-written helper
src/agentcore_cli.py - see the README for the prerequisites (Node.js 20+, uv).
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  SHARED HELPERS FOR THE TOOLS BELOW
# ═══════════════════════════════════════════════════════

# Return windows per customer tier, in days (project specification).
RETURN_WINDOW_DAYS = {'Standard': 30, 'Premium': 60}

# Order statuses for which a return can no longer be started.
_RETURN_ALREADY_STARTED = ('return_initiated', 'returned', 'refunded')


def _to_plain(value):
    """
    Recursively convert DynamoDB values into JSON-friendly Python types.

    boto3 returns every number as decimal.Decimal, which json.dumps cannot
    serialize (Strands would fall back to an unreadable repr). Integers
    become int, everything else float.

    Args:
        value: Any value returned by a DynamoDB read (dict, list, Decimal, str, ...)

    Returns:
        The same structure with Decimals replaced by int / float
    """
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


def _days_since(iso_date: str) -> Optional[int]:
    """
    Number of whole days between an ISO date (YYYY-MM-DD) and today.

    Args:
        iso_date: Date string as stored in the Orders table (order_date)

    Returns:
        Age in days, or None when the date is missing or malformed
    """
    try:
        return (date.today() - datetime.strptime(str(iso_date)[:10], '%Y-%m-%d').date()).days
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    # Worker model (Claude Sonnet 4.5) - low temperature for accurate fact reporting
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the InventoryAgent for NovaMart customer support.

ROLE: You are a DATA GATHERER. You look up order and customer facts in
NovaMart's systems and report them accurately. You NEVER make decisions.

TOOLS:
- check_order_status(customer_id, order_id): one specific order
- get_customer_tier(customer_id): the customer's tier (Standard / Premium) and profile
- list_customer_orders(customer_id): every order the customer has placed

PROCESS:
1. Always call get_customer_tier for the customer - downstream agents need the tier.
2. If the request names an order ID, call check_order_status with BOTH the
   customer_id and the order_id.
3. If no order ID is given but the request is about orders (e.g. "my recent
   order", "my headphones"), call list_customer_orders and identify the
   matching order(s) by product name or date.
4. Report the facts as a short, structured summary:
   customer_id, customer name, tier, and for each relevant order:
   order_id, product_name, status, order_date, days_since_order, price, quantity.

RULES:
- Report ONLY what the tools return. Never guess, invent or estimate data.
- If an order or customer is not found, say so plainly.
- NEVER decide or state whether an order is eligible for a return or refund,
  and never promise anything to the customer - eligibility is the RefundAgent's job.
- Do not write a customer-facing message; your output is read by other agents."""

    # NOTE: the Orders table has a COMPOSITE key (customer_id = partition key,
    # order_id = sort key), so a get_item needs BOTH values. That is why this
    # tool takes customer_id as well as order_id.
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price,
            quantity, days_since_order, ...) with found=True, or found=False
            and a not-found message
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        try:
            item = table.get_item(
                Key={'customer_id': customer_id.strip(), 'order_id': order_id.strip()}
            ).get('Item')
        except ClientError as exc:
            return {'found': False, 'error': f"Order lookup failed: {exc}"}

        if not item:
            return {
                'found': False,
                'customer_id': customer_id,
                'order_id': order_id,
                'message': f"No order {order_id} was found for customer {customer_id}.",
            }

        order = _to_plain(item)
        order['found'] = True
        order['days_since_order'] = _days_since(order.get('order_date'))
        order['checked_on'] = date.today().isoformat()
        return order

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile (customer_id, name, tier, account_created,
            total_orders, preferred_contact) with found=True, or found=False
            and a not-found message
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        try:
            item = table.get_item(Key={'customer_id': customer_id.strip()}).get('Item')
        except ClientError as exc:
            return {'found': False, 'error': f"Customer lookup failed: {exc}"}

        if not item:
            return {'found': False, 'customer_id': customer_id,
                    'message': f"No customer {customer_id} was found."}

        customer = _to_plain(item)
        # Contact details are not needed downstream - keep PII out of agent context.
        customer.pop('email', None)
        customer.pop('phone', None)
        customer['found'] = True
        return customer

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Dict with customer_id, order_count and an `orders` list, each entry
            holding order_id, product_name, status, order_date, days_since_order,
            price and quantity (newest order first)
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        items, query_args = [], {'KeyConditionExpression': Key('customer_id').eq(customer_id.strip())}
        try:
            while True:
                page = table.query(**query_args)
                items.extend(page.get('Items', []))
                if 'LastEvaluatedKey' not in page:
                    break
                query_args['ExclusiveStartKey'] = page['LastEvaluatedKey']
        except ClientError as exc:
            return {'customer_id': customer_id, 'error': f"Order listing failed: {exc}"}

        orders = []
        for item in _to_plain(items):
            orders.append({
                'order_id':         item.get('order_id'),
                'product_name':     item.get('product_name'),
                'status':           item.get('status'),
                'order_date':       item.get('order_date'),
                'days_since_order': _days_since(item.get('order_date')),
                'price':            item.get('price'),
                'quantity':         item.get('quantity'),
            })
        orders.sort(key=lambda o: o.get('order_date') or '', reverse=True)
        return {'customer_id': customer_id, 'order_count': len(orders), 'orders': orders}

    return Agent(
        name='InventoryAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    # Worker model (Claude Sonnet 4.5) - low temperature for consistent decisions
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1,
        region_name=config.AWS_REGION,
    )

    system_prompt = f"""You are the RefundAgent for NovaMart customer support.

ROLE: You decide whether an order is eligible for a return / refund and, when it
is eligible and the customer asked for it, you start the return.

RETURN POLICY (apply exactly):
- Standard tier customers: {RETURN_WINDOW_DAYS['Standard']}-day return window
- Premium tier customers:  {RETURN_WINDOW_DAYS['Premium']}-day return window
- The window is measured in days since the order date (days_since_order).
- Only orders with status "delivered" can be returned. Orders that are
  "processing" or "shipped" have not arrived yet; "cancelled" orders have
  nothing to return; orders already "return_initiated" must not be started twice.

DECISION PROCESS - follow these steps in order, every time:
1. ALWAYS call get_inventory_context(session_id) FIRST to read the facts the
   InventoryAgent gathered (customer tier, order status, order date, days_since_order).
   Never decide from memory or from the request text alone.
2. Identify the customer's tier and the order in question.
3. Look up the window for that tier (30 days Standard, 60 days Premium) and
   compare it with days_since_order.
4. Decide: ELIGIBLE or NOT ELIGIBLE, and why.
5. Only if the order is ELIGIBLE and the customer actually asked to return it or
   get a refund, call initiate_refund(customer_id, order_id, reason). Use the
   customer's stated reason, or "Customer requested return" if none was given.
   If they only asked about status or eligibility, do NOT initiate anything.
6. If the inventory context is missing or the order was not found, do not guess -
   report that the decision could not be made and what is missing.

OUTPUT: a short structured summary for the CommunicationAgent containing:
decision (ELIGIBLE / NOT ELIGIBLE / NOT DETERMINED), order_id, product, tier,
window_days, days_since_order, the reason, and - if a return was initiated - the
return_reference, refund_amount and the next-step instructions returned by the tool.
Do not write the customer-facing message yourself."""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            Dict with session_id, customer_id and the inventory_agent findings
            from WorkflowState, or an empty dict if the session or the findings
            do not exist yet
        """
        state = _read_workflow_state(session_id.strip())
        if not state or not state.get('inventory_agent'):
            return {}
        return {
            'session_id':      state.get('session_id'),
            'customer_id':     state.get('customer_id'),
            'inventory_agent': state.get('inventory_agent'),
        }

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        The tier window is re-checked against the stored order before anything
        is written, so an ineligible order can never be marked for return.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with status RETURN_INITIATED, return_reference,
            refund_amount and instructions - or status NOT_ELIGIBLE /
            ALREADY_INITIATED / ERROR with an explanation
        """
        customer_id, order_id = customer_id.strip(), order_id.strip()
        orders_table    = dynamodb.Table(config.ORDERS_TABLE)
        customers_table = dynamodb.Table(config.CUSTOMERS_TABLE)

        try:
            order    = orders_table.get_item(Key={'customer_id': customer_id, 'order_id': order_id}).get('Item')
            customer = customers_table.get_item(Key={'customer_id': customer_id}).get('Item')
        except ClientError as exc:
            return {'status': 'ERROR', 'message': f"Could not read order/customer: {exc}"}

        if not order:
            return {'status': 'ERROR', 'message': f"Order {order_id} not found for customer {customer_id}."}

        order_status = str(order.get('status', '')).lower()
        if order_status in _RETURN_ALREADY_STARTED:
            return {
                'status':           'ALREADY_INITIATED',
                'order_id':         order_id,
                'return_reference': order.get('return_reference'),
                'message':          f"A return for {order_id} was already started "
                                    f"(order status: {order_status}).",
            }

        tier        = (customer or {}).get('tier', 'Standard')
        window_days = RETURN_WINDOW_DAYS.get(tier, RETURN_WINDOW_DAYS['Standard'])
        days_since  = _days_since(order.get('order_date'))

        if order_status != 'delivered':
            return {'status': 'NOT_ELIGIBLE', 'order_id': order_id, 'tier': tier,
                    'message': f"Only delivered orders can be returned (status: {order_status})."}
        if days_since is None or days_since > window_days:
            return {'status': 'NOT_ELIGIBLE', 'order_id': order_id, 'tier': tier,
                    'window_days': window_days, 'days_since_order': days_since,
                    'message': f"Order is {days_since} days old; the {tier} return window "
                               f"is {window_days} days."}

        refund_amount    = Decimal(str(order.get('price', '0'))) * int(order.get('quantity', 1) or 1)
        return_reference = f"RET-{order_id.split('-')[-1]}-{uuid.uuid4().hex[:6].upper()}"
        initiated_at     = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        try:
            # Conditional write: the order must still exist and still be "delivered",
            # so two concurrent requests cannot start the same return twice.
            orders_table.update_item(
                Key={'customer_id': customer_id, 'order_id': order_id},
                UpdateExpression=('SET #st = :new_status, return_reference = :ref, '
                                  'return_reason = :reason, return_initiated_at = :ts, '
                                  'refund_amount = :amount'),
                ConditionExpression='attribute_exists(order_id) AND #st = :delivered',
                ExpressionAttributeNames={'#st': 'status'},
                ExpressionAttributeValues={
                    ':new_status': 'return_initiated',
                    ':ref':        return_reference,
                    ':reason':     reason or 'Customer requested return',
                    ':ts':         initiated_at,
                    ':amount':     str(refund_amount),
                    ':delivered':  'delivered',
                },
            )
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            return {'status': 'ALREADY_INITIATED', 'order_id': order_id,
                    'message': f"Order {order_id} changed while processing - a return may already exist."}
        except ClientError as exc:
            return {'status': 'ERROR', 'message': f"Could not update order {order_id}: {exc}"}

        return {
            'status':           'RETURN_INITIATED',
            'order_id':         order_id,
            'product_name':     order.get('product_name'),
            'tier':             tier,
            'window_days':      window_days,
            'days_since_order': days_since,
            'return_reference': return_reference,
            'refund_amount':    f"${refund_amount:.2f}",
            'initiated_at':     initiated_at,
            'instructions':     ("A prepaid return shipping label will be emailed to the customer. "
                                 "Pack the item with all original accessories and drop it off with "
                                 "the carrier. The refund goes to the original payment method within "
                                 "5-7 business days of NovaMart receiving the return."),
        }

    return Agent(
        name='RefundAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    def _retriever_prompt(domain: str, tool_name: str) -> str:
        """System prompt shared by the three retriever sub-agents."""
        return f"""You are the {domain}PolicyRetrieverAgent for NovaMart.
You retrieve passages from the {domain} Policy knowledge base - nothing else.

1. Call {tool_name} exactly once, passing the user's question as the query.
2. Return the retrieved passages faithfully: keep each passage's wording and its
   Source line. You may drop passages that are clearly unrelated to the question.
3. Do not add knowledge, opinions or answers that are not in the passages.
4. If nothing relevant is returned, reply exactly:
   "No relevant {domain.lower()} policy passages found.\""""

    # Retriever sub-agents: worker model, temperature 0.0 for deterministic retrieval
    def _retriever_model() -> BedrockModel:
        return BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            temperature=0.0,
            region_name=config.AWS_REGION,
        )

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Returns Policy knowledge base.

        Args:
            query: The policy question to search for

        Returns:
            Formatted passages (text, relevance score, S3 source) from the Returns KB
        """
        return format_kb_results(retrieve_from_knowledge_base(config.RETURNS_KB_ID, query))

    returns_retriever = Agent(
        name='ReturnsPolicyRetrieverAgent',
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Returns', 'retrieve_returns_policy'),
        tools=[retrieve_returns_policy],
    )

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Shipping Policy knowledge base.

        Args:
            query: The policy question to search for

        Returns:
            Formatted passages (text, relevance score, S3 source) from the Shipping KB
        """
        return format_kb_results(retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query))

    shipping_retriever = Agent(
        name='ShippingPolicyRetrieverAgent',
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Shipping', 'retrieve_shipping_policy'),
        tools=[retrieve_shipping_policy],
    )

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """
        Retrieve relevant passages from the Warranty Policy knowledge base.

        Args:
            query: The policy question to search for

        Returns:
            Formatted passages (text, relevance score, S3 source) from the Warranty KB
        """
        return format_kb_results(retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query))

    warranty_retriever = Agent(
        name='WarrantyPolicyRetrieverAgent',
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Warranty', 'retrieve_warranty_policy'),
        tools=[retrieve_warranty_policy],
    )

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases, one
            labelled section per domain (Returns, Shipping, Warranty)
        """
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            This covers both the direct worker thread and any internal
            streaming child threads that Strands SDK spawns internally -
            which do NOT inherit thread-local variables and therefore cannot
            be suppressed with a thread-local capture approach.
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            try:
                agent.messages = []          # each query starts from a clean context
                return domain, str(agent(query)).strip()
            except Exception as exc:         # one failed KB must not sink the others
                logger.exception("%s retriever failed", domain)
                return domain, f"[{domain} retriever error: {exc}]"

        results = {}
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(_run_retriever, domain, agent, query): domain
                    for domain, agent in retrievers.items()
                }
                for future in as_completed(futures):
                    domain, text = future.result()
                    results[domain] = text
        finally:
            # ── Trace: all KBs responded - always clears stdout suppression ──
            trace.kb_done(len(retrievers))

        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        return "\n\n".join(
            f"=== {domain.upper()} POLICY (from {domain}PolicyRetrieverAgent) ===\n"
            f"{results.get(domain) or '[No results]'}"
            for domain in ['Returns', 'Shipping', 'Warranty']
        )

    # Coordinator: worker model, temperature 0.2 for a natural but grounded synthesis
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.2,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the PolicyAgent for NovaMart customer support.
You answer questions about NovaMart's policies (returns, shipping, warranty,
customer tiers) using ONLY passages retrieved from the policy knowledge bases.

PROCESS:
1. ALWAYS call search_all_policies FIRST with the question. It queries the
   Returns, Shipping and Warranty knowledge bases in parallel. Never answer
   before calling it, and never answer from general knowledge.
2. Read all three sections and synthesize ONE grounded answer:
   - include every relevant rule, number, time window and tier difference
     (e.g. Standard vs Premium);
   - combine information across domains when the question spans them;
   - mention which policy (Returns / Shipping / Warranty) each fact comes from.
3. If the passages do not cover the question, say so clearly - do not guess.

OUTPUT: a concise, factual policy summary for the CommunicationAgent (not a
customer-facing letter). You have no access to customer accounts or orders."""

    return Agent(
        name='PolicyAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    # Worker model (Claude Sonnet 4.5) - temperature 0.3 for a warm, natural tone
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.3,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the CommunicationAgent for NovaMart customer support.
You write the FINAL message the customer reads.

PROCESS:
1. ALWAYS call get_full_workflow_context(session_id) first. It holds everything
   the other agents found for this session:
   - inventory_agent: order and customer facts
   - refund_agent: the return / refund decision (and return reference, if any)
   - policy_agent: the relevant policy rules from the knowledge bases
   Columns that are missing simply were not needed for this request.
2. Answer the customer's CURRENT request (given to you in the prompt). Use
   earlier findings in the context only when they are relevant to it.
3. Include ALL relevant information from the previous agents - never drop a
   decision, a reason, a number, a return reference, a refund amount, a time
   frame or a next step that the customer needs.
4. Use only facts from the workflow context or the request. Never invent order
   details, policies, references, dates or promises.

SPECIAL CASES:
- Return approved: confirm it, give the return reference, refund amount,
  and the next steps / refund timeline.
- Return not eligible: explain why kindly (tier window vs. order age, or order
  status) and, if the context supports it, mention alternatives such as the
  warranty for defective items.
- Policy question: explain the policy clearly, including tier differences.
- Calculation questions: compute the answer yourself, accurately, step by step.
  Keep full precision through the calculation and round currency to cents only
  at the end (e.g. 5 x $29.99 = $149.95; 10% off = $14.995; total $134.955
  -> $134.96). Show the short working.
- If information is missing or an agent reported an error, apologise, say what
  you could not confirm, and offer to help further.

STYLE: warm, empathetic and professional; address the customer by first name
if it is known; plain text (short paragraphs or bullets), no XML or markdown
headings; concise. Sign off as "NovaMart Customer Support". Never ask for
card numbers, passwords or other sensitive data."""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (session_id, customer_id, version and the
            inventory_agent, policy_agent, refund_agent findings that exist),
            or a dict with an `error` key if the session does not exist
        """
        state = _read_workflow_state(session_id.strip())
        if not state:
            return {'error': f"No WorkflowState found for session {session_id}."}
        context = _to_plain(state)
        context.pop('ttl', None)
        context.pop('communication_agent', None)   # previous drafts are not inputs
        return context

    return Agent(
        name='CommunicationAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    # Orchestrator model (Claude Haiku 4.5) - temperature 0.0 for deterministic routing
    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        temperature=0.0,
        region_name=config.AWS_REGION,
    )

    system_prompt = """You are the OrchestratorAgent for NovaMart customer support.
You ROUTE requests to specialist agents and manage the shared WorkflowState.
You never answer the customer yourself.

Every message starts with "[Session ID: <id>] [Customer ID: <id>]" followed by
the customer's request. Use exactly those IDs in every tool call, and pass the
customer's request text unchanged to the specialists.

ROUTING RULES - follow them in this order, with no exceptions:

RULE 1 - EVERY request: call initialize_session(session_id, customer_id) FIRST.

RULE 2 - Order status, return or refund requests (e.g. "where is my order",
  "I want to return ORD-123", "can I get a refund"): call
  route_to_inventory_agent FIRST, THEN route_to_refund_agent.
  Never call the refund agent before the inventory agent.

RULE 3 - Policy meaning questions (return windows, shipping rates or times,
  warranty terms, what a tier includes in general): call route_to_policy_agent.

RULE 4 - Account questions about THIS customer ("what is my tier?",
  "am I premium?", "what orders do I have?"): call route_to_inventory_agent.
  NEVER send account questions to the policy agent - it only knows policy text,
  not customer data. Do not call the refund agent unless a return/refund is asked for.

RULE 5 - Math / calculation questions (prices, discounts, totals): call NO
  specialist (skip inventory, policy and refund). Go straight to
  route_to_communication_agent, which performs the calculation.

RULE 6 - EVERY request: call route_to_communication_agent LAST, exactly once,
  after all other routing calls have returned. It writes the reply.

A request may need several rules (e.g. a return plus a policy question):
apply each relevant one, then finish with Rule 6.

CRITICAL:
- You must NEVER write the customer-facing response yourself, not even a
  short one, and never do the math yourself. The CommunicationAgent ALWAYS
  composes the final response - no exceptions.
- After route_to_communication_agent returns, reply with exactly the text it
  returned, unchanged, and nothing else.
- Do not answer from your own knowledge and do not skip initialize_session."""

    def _state_version(session_id: str, customer_id: str = 'UNKNOWN') -> int:
        """Read the current WorkflowState version, creating the record if it is missing."""
        state = _read_workflow_state(session_id)
        if state is None:
            state = _create_workflow_state(session_id, customer_id)
        return int(state['version'])

    def _invoke_worker(worker: Agent, prompt: str) -> str:
        """Run a worker agent on a fresh conversation and return its text (errors as text)."""
        try:
            worker.messages = []     # WorkflowState is the shared memory, not chat history
            return str(worker(prompt)).strip()
        except Exception as exc:     # report, so the CommunicationAgent can still reply
            logger.exception("Worker agent %s failed", getattr(worker, 'name', worker))
            return f"[{getattr(worker, 'name', 'worker')} error: {exc}]"

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        # 1. read the current WorkflowState
        expected_version = _state_version(session_id, customer_id)
        trace.step_start('inventory_agent')
        trace.agent_section('INVENTORY AGENT')

        # 2. invoke the worker
        result = _invoke_worker(inventory_agent, (
            f"Session ID: {session_id}\nCustomer ID: {customer_id}\n"
            f"Customer request: {request}\n\n"
            "Gather the customer's tier and all order facts relevant to this request."
        ))

        # 3. write the result back with optimistic locking
        _update_workflow_state(session_id, {'inventory_agent': result}, expected_version)
        trace.step_done('inventory_agent', expected_version)
        return result

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        expected_version = _state_version(session_id)
        trace.step_start('policy_agent')
        trace.agent_section('POLICY AGENT')

        result = _invoke_worker(policy_agent, request)

        _update_workflow_state(session_id, {'policy_agent': result}, expected_version)
        trace.step_done('policy_agent', expected_version)
        return result

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        expected_version = _state_version(session_id, customer_id)
        trace.step_start('refund_agent')
        trace.agent_section('REFUND AGENT')

        result = _invoke_worker(refund_agent, (
            f"Session ID: {session_id}\nCustomer ID: {customer_id}\n"
            f"Customer request: {request}\n\n"
            "Read the inventory context for this session, decide eligibility, and "
            "initiate the return only if it is eligible and was requested."
        ))

        _update_workflow_state(session_id, {'refund_agent': result}, expected_version)
        trace.step_done('refund_agent', expected_version)
        return result

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        expected_version = _state_version(session_id, customer_id)
        trace.step_start('communication_agent')
        trace.agent_section('COMMUNICATION AGENT')

        result = _invoke_worker(communication_agent, (
            f"Session ID: {session_id}\nCustomer ID: {customer_id}\n"
            f"Customer's current request: {original_request}\n\n"
            "Read the full workflow context and write the final customer-facing reply."
        ))

        _update_workflow_state(session_id, {'communication_agent': result}, expected_version)
        trace.step_done('communication_agent', expected_version)
        return result

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized (or that an existing
            session for a follow-up message was found), including its version
        """
        try:
            state = _create_workflow_state(session_id, customer_id)
            return (f"Session {session_id} initialized for customer {customer_id} "
                    f"(WorkflowState version {state['version']}).")
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            # Follow-up message in the same chat session - keep the existing record.
            state = _read_workflow_state(session_id) or {}
            return (f"Session {session_id} already exists for customer "
                    f"{state.get('customer_id', customer_id)} (WorkflowState version "
                    f"{int(state.get('version', 0))}); continuing it.")

    return Agent(
        name='OrchestratorAgent',
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT TOOLING - AgentCore CLI
#
#  The runtime is deployed with the AgentCore CLI (`agentcore`, npm package
#  @aws/agentcore - https://github.com/aws/agentcore-cli) through the
#  pre-written helper src/agentcore_cli.py:
#
#    agentcore_cli.stage_runtime_code()   copies this file, its helper modules
#                                         and config.py to build/runtime/ with a
#                                         pyproject.toml of the runtime deps
#    agentcore_cli.configure_runtime()    writes the runtime settings (network
#                                         mode, protocol, execution role, env
#                                         vars) to agentcore/agentcore.json
#    agentcore_cli.deploy()               runs `agentcore deploy -y`: the CLI
#                                         downloads arm64 / Python 3.12 wheels
#                                         with uv, zips them with the code
#                                         (direct code deployment) and creates
#                                         or updates the runtime via CDK
#    agentcore_cli.deployed_runtime_arn() reads the ARN the CLI recorded
#
#  Inside the runtime this same file is the entry point: it is started with
#  no command-line argument and serves HTTP (see run_serve). The marker file
#  written next to it by stage_runtime_code() tells __main__ to do so.
# ═══════════════════════════════════════════════════════

_RUNTIME_MARKER = '.agentcore-runtime'         # written by agentcore_cli.stage_runtime_code()
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    # TODO: Create the guardrail
    # Use bedrock_client.create_guardrail() with:
    #   - name (config.GUARDRAIL_NAME) and description
    #   - contentPolicyConfig - filtersConfig for SEXUAL, VIOLENCE, HATE at HIGH
    #     strength and INSULTS, MISCONDUCT at MEDIUM strength (input + output)
    #   - sensitiveInformationPolicyConfig - piiEntitiesConfig:
    #       CREDIT_DEBIT_CARD_NUMBER and US_SOCIAL_SECURITY_NUMBER -> BLOCK
    #       EMAIL and PHONE -> ANONYMIZE
    #   - topicPolicyConfig - one DENY topic per entry in config.GUARDRAIL_BLOCKED_TOPICS
    #     (competitor products, pricing negotiations, legal threats)
    #     Use topicPolicyConfig.tierConfig = {'tierName': 'STANDARD'} and
    #     top-level crossRegionConfig = {'guardrailProfileIdentifier': 'us.guardrail.v1:0'}.
    #     Define pricing negotiations as haggling / changing an advertised price,
    #     excluding arithmetic using an already-specified price and discount.
    #     Classic-tier definitions tested in this project blocked the math scenario.
    #     Validate allowed arithmetic (input and output) and blocked negotiation,
    #     competitor and legal-threat requests. Keep all required safety policies.
    #   - wordPolicyConfig - managedWordListsConfig with type PROFANITY
    #   - blockedInputMessaging and blockedOutputsMessaging
    #
    # Then promote it from DRAFT to a numbered version with
    # bedrock_client.create_guardrail_version(guardrailIdentifier=...)
    # and return (guardrail_id, guardrail_version).

    # One DENY topic per entry in config.GUARDRAIL_BLOCKED_TOPICS.
    # Definitions stay under 200 characters; examples under 100 characters each.
    topic_specs = {
        'competitor products': {
            'definition': ("Requests to discuss, recommend, compare or price-match products, stores "
                           "or services of other retailers or competitors instead of NovaMart."),
            'examples': [
                "Is Amazon's version of these headphones better than yours?",
                "Which competitor sells this laptop for less?",
                "Should I buy this from Best Buy or Walmart instead?",
                "Can you recommend a store other than NovaMart for this?",
            ],
        },
        'pricing negotiations': {
            'definition': ("Haggling over price, or asking NovaMart to lower, match, waive or "
                           "otherwise change an advertised price. Calculating a total from a "
                           "stated price and discount is not negotiation."),
            'examples': [
                "Can you give me a better price on this laptop?",
                "I'll buy it today if you knock $50 off.",
                "Lower the price to $20 and we have a deal.",
                "Match the price I saw elsewhere or I'm not buying.",
                "Give me an extra discount on top of the sale price.",
            ],
        },
        'legal threats': {
            'definition': ("Threats of lawsuits, attorneys, court action, regulatory complaints "
                           "or other legal action against NovaMart or its staff."),
            'examples': [
                "Refund me now or I will sue NovaMart.",
                "My lawyer will be contacting you about this.",
                "I'm taking you to small claims court.",
                "I will file a legal complaint against your company.",
            ],
        },
    }
    topics_config = []
    for topic_name in config.GUARDRAIL_BLOCKED_TOPICS:
        spec = topic_specs[topic_name]
        topics_config.append({
            'name':       topic_name.title(),
            'definition': spec['definition'],
            'examples':   spec['examples'],
            'type':       'DENY',
        })

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description=("NovaMart customer-support safety guardrail: harmful content, PII, "
                     "competitor / pricing-negotiation / legal-threat topics and profanity."),
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL',     'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE',   'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'HATE',       'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
                {'type': 'INSULTS',    'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            ],
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER',  'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'EMAIL',                     'action': 'ANONYMIZE'},
                {'type': 'PHONE',                     'action': 'ANONYMIZE'},
            ],
        },
        topicPolicyConfig={
            'topicsConfig': topics_config,
            'tierConfig':   {'tierName': 'STANDARD'},
        },
        wordPolicyConfig={
            'managedWordListsConfig': [{'type': 'PROFANITY'}],
        },
        crossRegionConfig={'guardrailProfileIdentifier': 'us.guardrail.v1:0'},
        blockedInputMessaging=(
            "I'm sorry, but I can't help with that request. I'm happy to help with your "
            "NovaMart orders, returns, refunds, shipping or warranty questions - please "
            "don't include payment card or Social Security numbers in your message."
        ),
        blockedOutputsMessaging=(
            "I'm sorry, but I can't share that response. Please contact NovaMart Customer "
            "Support and we'll be glad to help with your order, return or policy question."
        ),
    )
    guardrail_id = response['guardrailId']
    print(f"  Guardrail created: {guardrail_id} (DRAFT)")

    # Publish an immutable numbered version - agents must never run against DRAFT.
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Initial published version for the NovaMart multi-agent system",
    )
    guardrail_version = version_response['version']

    # The new version is CREATING for a few seconds; wait until it is usable.
    deadline = time.time() + 120
    while time.time() < deadline:
        status = bedrock_client.get_guardrail(
            guardrailIdentifier=guardrail_id, guardrailVersion=guardrail_version
        ).get('status')
        if status == 'READY':
            break
        if status == 'FAILED':
            raise RuntimeError(f"Guardrail version {guardrail_version} failed to publish")
        time.sleep(3)

    print(f"  Guardrail version published: {guardrail_version}")
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime with
    the AgentCore CLI (src/agentcore_cli.py wraps it).

    AgentCore does not serialize Python objects, so `orchestrator_agent` is
    not uploaded directly. Instead the pre-written staging step copies this
    file, which doubles as the HTTP entry point (see run_serve), together
    with its helper modules and config.py to build/runtime/. `agentcore
    deploy` then packages that directory with arm64 dependencies and creates
    or updates the runtime ("direct code deployment"). Re-running is safe:
    an unchanged runtime is left alone, a changed one is updated in place.

    The guardrail is attached by environment variables: inside the runtime
    build_agent_graph() reads GUARDRAIL_ID / GUARDRAIL_VERSION and applies
    them to every agent's model (see _apply_guardrail), exactly as `test`
    and `chat` do locally.

    Returns:
        The AgentCore Runtime ARN
    """
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME
    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")
    print(f"  Runtime: {runtime_name}  |  CLI project: agentcore/agentcore.json "
          f"(stack {agentcore_cli.stack_name()})")
    previous_arn = agentcore_cli.deployed_runtime_arn()
    if previous_arn:
        print(f"  Runtime already deployed - updating it: {previous_arn}")

    # Stage the code the CLI packages (src modules + config.py + pyproject.toml).
    agentcore_cli.stage_runtime_code()

    # TODO: Configure and deploy the runtime with the AgentCore CLI
    # 1. Build the runtime environment variables dict `runtime_env` with:
    #      AWS_REGION, PROJECT_NAME (config.AWS_REGION / config.PROJECT_NAME),
    #      RETURNS_KB_ID, SHIPPING_KB_ID, WARRANTY_KB_ID (from config),
    #      AGENT_LOG_GROUP (config.AGENT_LOG_GROUP), and the guardrail
    #      (GUARDRAIL_ID = guardrail_id, GUARDRAIL_VERSION = guardrail_version)
    # 2. Write the runtime settings to agentcore/agentcore.json with
    #      agentcore_cli.configure_runtime(env_vars=runtime_env,
    #                                      network_mode='PUBLIC',
    #                                      protocol='HTTP',
    #                                      execution_role_arn=config.AGENTCORE_ROLE_ARN)
    # 3. Deploy:  agentcore_cli.deploy()        (runs `agentcore deploy -y`)
    # 4. Read the ARN the CLI recorded:
    #      runtime_arn = agentcore_cli.deployed_runtime_arn()
    runtime_env = {
        'AWS_REGION':        config.AWS_REGION,
        'PROJECT_NAME':      config.PROJECT_NAME,
        'RETURNS_KB_ID':     config.RETURNS_KB_ID,
        'SHIPPING_KB_ID':    config.SHIPPING_KB_ID,
        'WARRANTY_KB_ID':    config.WARRANTY_KB_ID,
        'AGENT_LOG_GROUP':   config.AGENT_LOG_GROUP,
        'GUARDRAIL_ID':      guardrail_id,
        'GUARDRAIL_VERSION': str(guardrail_version),
    }
    missing = [key for key, value in runtime_env.items() if not value]
    if missing:
        print(f"  [Warning] Empty runtime environment variables (skipped by the CLI): {missing}")
    if str(guardrail_version).upper() == 'DRAFT':
        print("  [Warning] GUARDRAIL_VERSION is DRAFT - publish a numbered version first")

    agentcore_cli.configure_runtime(
        env_vars=runtime_env,
        network_mode='PUBLIC',
        protocol='HTTP',
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )
    agentcore_cli.deploy()
    runtime_arn = agentcore_cli.deployed_runtime_arn()

    if not runtime_arn:
        raise NotImplementedError("deploy_to_agentcore_runtime: AgentCore CLI deployment not implemented")

    # Wait for the runtime to become READY and return its ARN.
    print(f"  Runtime deployed: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, runtime_arn.split('/')[-1])
    print(' ready.')
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    # TODO: Create AgentCore Memory
    # Use agentcore_control.create_memory() with:
    #   - name (memory_name) and a description
    #   - eventExpiryDuration = 7   (days)
    #   - memoryStrategies = [{'summaryMemoryStrategy': {
    #         'name': 'SessionSummary',
    #         'namespaces': ['/summaries/{actorId}/{sessionId}']}}]
    #   - clientToken (e.g. str(uuid.uuid4())) for idempotency
    # Store the API response in `response`.
    response = agentcore_control.create_memory(
        name=memory_name,
        description=("Session-scoped conversational memory for the NovaMart multi-agent "
                     "customer support system (session summaries, 7-day event retention)."),
        eventExpiryDuration=7,
        memoryStrategies=[{
            'summaryMemoryStrategy': {
                'name':        'SessionSummary',
                'description': 'Summarizes each customer support session',
                'namespaces':  ['/summaries/{actorId}/{sessionId}'],
            }
        }],
        clientToken=str(uuid.uuid4()),
    )

    if response is None:
        raise NotImplementedError("configure_memory: create_memory() not implemented")

    # Wait until the memory resource is ACTIVE and return its ARN.
    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    # TODO: Build the logging configuration
    # logging_configuration = {
    #     'cloudWatchConfig': {'logGroupName': config.AGENT_LOG_GROUP,
    #                          'logLevel': 'INFO', 'enabled': True},
    #     'xRayConfig':       {'enabled': True, 'samplingRate': 1.0},
    # }
    # Then apply it:  summary = apply_observability_config(runtime_arn, logging_configuration)
    # Wrap the call in try/except - on success print the CloudWatch log group
    # and the X-Ray sampling rate; on exception print
    #   "[Note] Observability configuration failed: <e>"

    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel':     'INFO',
            'enabled':      True,
        },
        'xRayConfig': {
            'enabled':      True,
            'samplingRate': 1.0,
        },
    }

    try:
        summary = apply_observability_config(runtime_arn, logging_configuration)
        print(f"  CloudWatch Logs : {summary.get('log_group', config.AGENT_LOG_GROUP)} (INFO)")
        print(f"  X-Ray tracing   : enabled, sampling rate "
              f"{logging_configuration['xRayConfig']['samplingRate']:.0%}")
    except Exception as e:
        print(f"  [Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION = os.environ.get('ORDERS_FUNCTION', '')
_POLICY_FUNCTION = os.environ.get('POLICY_FUNCTION', '')
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', '')


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Optional extension to the in-process @tool functions. Resolve configured
    Lambda functions first; if none exist, skip gateway creation. Otherwise
    create/reuse the gateway and submit its targets. Connecting agents to this
    MCP endpoint requires separate integration; this starter uses in-process tools.

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        A SKIPPED result with a reason, or gateway details and target count.
    """
    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    # Resolve optional Lambda targets before creating any gateway resources.
    available = []
    for target in targets:
        if not target['function']:
            continue
        try:
            available.append((target, _gw_get_function_arn(target['function'])))
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ResourceNotFoundException':
                raise
            print(f"    [Skipped] {target['name']}: Lambda function not found")

    if not available:
        return {'status': 'SKIPPED', 'reason': 'No configured Lambda tool functions are available.'}

    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    print(f"\n  Registering {len(available)} Gateway targets...")
    for target, lambda_arn in available:
        _gw_create_target(agentcore_ctrl, gateway_id, target, lambda_arn)

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url,
            'status': 'TARGETS_SUBMITTED', 'target_count': len(available)}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    # Fail fast if the AgentCore CLI (used by Steps 3 and 5) is missing.
    import agentcore_cli
    print(f"AgentCore CLI: {agentcore_cli.cli_version()} ({agentcore_cli.cli_path()})\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        if gw['status'] == 'SKIPPED':
            print(f"  [Skipped] Gateway: {gw['reason']}")
        else:
            print(f"  Gateway URL : {gw['gateway_url']}")
            print("  Lambda targets submitted; connect an MCP client separately to use them.")
    except Exception as e:
        print(f"  [Note] Optional Gateway deployment failed: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"")
    print(f"  or with the CLI:  agentcore invoke \"What is the return policy for premium customers?\"")
    print(f"  (agentcore status / agentcore logs show the deployed runtime and its logs)\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")
