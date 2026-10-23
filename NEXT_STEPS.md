# NovaMart: what's done, what's left

`src/agent_orchestrator.py` is complete: every TODO for Tasks 2, 3, 4 and 6 is implemented.
No other file was changed. Everything below needs your AWS account (region `us-east-1`).

## 1. Setup (once)
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Deploy `infrastructure/starter_stack.yaml` as stack **udacity-agentcore** (Console or CloudShell,
`CAPABILITY_NAMED_IAM`) and wait for `CREATE_COMPLETE`. Then:
```bash
python infrastructure/seed_data.py
copy .env.example .env           # macOS/Linux: cp .env.example .env
python config.py
python tests/test_agent.py task2 # expect 40/40
```
The deploy step also needs the AgentCore CLI (Node.js 20+, `npm i -g @aws/agentcore`, and `uv`).
See README.md.

## 2. Knowledge Bases (Console, Task 5)
Bedrock → Knowledge Bases → Create, using the **self-managed** option (NOT "Managed KB").

| Name | S3 prefix | Vector index |
|---|---|---|
| novamart-returns-policy-kb  | policies/returns/  | returns-policy-index  |
| novamart-shipping-policy-kb | policies/shipping/ | shipping-policy-index |
| novamart-warranty-policy-kb | policies/warranty/ | warranty-policy-index |

Embeddings: Titan Text Embeddings v2. Vector store: S3 Vectors → existing vector bucket (shown by
`python config.py`). Click **Sync** on each data source. Put the three IDs in `.env`, then:
```bash
python tests/test_agent.py task5
```

## 3. Deploy + verify (Tasks 3, 4, 6)
```bash
python src/agent_orchestrator.py deploy
```
Copy the printed `AGENTCORE_RUNTIME_ARN`, `GUARDRAIL_ID` and `GUARDRAIL_VERSION` into `.env`.
```bash
python src/agent_orchestrator.py test     # 3 scenarios, prints X-Ray trace ids
python tests/test_agent.py all            # screenshot: 120/120
```
Wait about 60 seconds, then go to CloudWatch → X-Ray traces → Service map → Last 5 minutes and
take the second screenshot.

## Guardrail spot checks (from the spec)
In `python src/agent_orchestrator.py chat`:
- "How much are 5 items at $29.99 with 10% off?" should be allowed. The answer is $134.96.
- "Can you knock $50 off this laptop?" should be blocked.
- "Is Amazon's version better than yours?" should be blocked.
- "Refund me or I'll sue NovaMart." should be blocked.

## Notes
- The ORD-27176 return can only be initiated once. A re-run reports that the return already
  exists, with its reference. To reset it, run `python infrastructure/seed_data.py` again.
- Clean up only after you submit: `python infrastructure/cleanup.py`, then add `--yes`.
