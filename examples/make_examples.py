"""Reproduce Apache-2.0 authored examples, with disjoint TRAIN/DEV wording.

The routing example teaches a fictional company's private ownership mapping.
Every authored ticket has two contract counterfactuals in the SAME split/group.
No customer message template or message string crosses the split boundary.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent


def write_jsonl(name, rows):
    (ROOT / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


TRAIN = {
    "revenue": [
        "Our renewal invoice still charges for the seats we removed last month.",
        "A single upgrade produced two card debits. Please reverse the duplicate payment.",
        "Can the next subscription payment be split between two cost centers?",
        "The tax number on our receipt belongs to our former legal entity.",
        "We canceled before the trial ended but received a subscription bill anyway.",
        "The agreed volume discount did not appear on this month's statement.",
        "Please explain why our annual plan was renewed at the monthly rate.",
        "Our purchase order requires a different billing address on the invoice.",
        "We prepaid for twelve months and need the unused credit calculated after cancellation.",
        "Could you switch our payment method from a credit card to bank transfer?",
        "The accounting team needs a credit note for an accidentally purchased extra license.",
        "A currency conversion surcharge appeared even though our contract is denominated in euros.",
        "We can sign in normally; the unresolved problem is the amount of sales tax charged.",
        "Yesterday's outage is over. This ticket is solely a request for its contractual service credit.",
        "The API works again, but we need an itemized explanation of the overage fee.",
        "Do not reset our passwords. Please refund the charge associated with the closed workspace.",
        "The dashboard is accessible; our finance director disputes the late-payment penalty.",
        "We are not reporting an integration bug. We want a quote for adding thirty users.",
        "A payment was taken after our nonprofit exemption had already been approved.",
        "Our invoice lists the wrong company registration details and cannot be reimbursed.",
        "We need a copy of the signed commercial rate schedule used for this charge.",
        "The receipt was sent to an old finance mailbox; please reissue it to our accounts team.",
        "The order form promises quarterly installments, but the entire annual amount was collected.",
        "Our license count is correct. The monetary rounding on the statement is not.",
    ],
    "identity": [
        "The password-reset message never arrives, so I cannot enter the workspace.",
        "My phone was replaced and I no longer have the authenticator used for sign-in.",
        "A new colleague's invitation expired before they accepted it. Please issue another.",
        "Company single sign-on loops back to the login screen for one employee.",
        "The only workspace owner left our organization; we need to transfer ownership.",
        "My account is locked following several mistyped passwords.",
        "Please revoke the active sessions belonging to a former contractor.",
        "An employee changed their surname and needs their account email updated.",
        "We need to enforce multifactor authentication for every workspace administrator.",
        "An invitation was accepted using a personal email instead of the intended work identity.",
        "Please remove the administrator role from someone who moved to another team.",
        "I lost the recovery codes and cannot complete the second authentication step.",
        "Our invoices are paid. The problem is that the account recovery link says it is invalid.",
        "The website is up for everyone else; only my user has been disabled.",
        "This is not an API question. I need permission to join an existing workspace.",
        "We do not need a refund. We need access restored to the person who manages payments.",
        "The service is responsive once logged in, but our directory user is mapped to the wrong account.",
        "Billing is accurate. Please delete the stale login identity left by our old domain.",
        "Our security administrator needs to rotate the workspace recovery contacts.",
        "A colleague can view projects but should have been granted an editor role.",
        "The invitation landed in the wrong tenant because two users share a similar name.",
        "Please disconnect our retired identity provider without locking out current owners.",
        "I accidentally created a second profile; help me recover the original organization membership.",
        "We need the list of devices currently signed into this administrator account.",
    ],
    "reliability": [
        "Every workspace user receives a 503 when they open the dashboard.",
        "Requests that normally finish in a second now take a minute in all projects.",
        "Scheduled jobs stopped running across the region despite no configuration changes.",
        "File uploads fail for everyone on several networks with server-side errors.",
        "Our dashboards have been unavailable for twenty minutes across the organization.",
        "The service times out simultaneously in two regions.",
        "Search results stopped updating yesterday even though new records are present.",
        "Background exports remain stuck in queued state for every tenant we manage.",
        "The notification delivery system is dropping messages for multiple customers.",
        "Graphs show stale values while the raw data ingestion endpoint continues to accept events.",
        "All users are seeing intermittent gateway failures during ordinary page loads.",
        "A maintenance deployment seems to have stopped scheduled report delivery.",
        "Passwords are accepted, but the shared service crashes immediately after the dashboard loads.",
        "This is not a pricing complaint. The metering service itself is returning internal errors.",
        "We know how to call the API; production calls now fail for every client without code changes.",
        "Our account permissions are correct. Project pages nevertheless return server exceptions.",
        "Do not send an invoice copy. Please restore the unavailable invoice-rendering service.",
        "The integration was unchanged for months; sudden elevated latency now affects all endpoints.",
        "Record synchronization has fallen several hours behind for every workspace in this zone.",
        "A batch-processing worker pool is repeatedly restarting and leaving tasks unfinished.",
        "The status endpoint is green, but all attempts to save documents end in gateway timeouts.",
        "Analytics queries are suddenly timing out even for previously fast small datasets.",
        "We see a sharp rise in service-side errors after today's release.",
        "The platform is discarding completed uploads before they appear in the file list.",
    ],
    "developer": [
        "Which SDK method lists projects using cursor-based pagination?",
        "I am building a webhook receiver. Which signature must I verify before accepting an event?",
        "Please show how to call the API using a service-account token from Python.",
        "How should a nested filter be represented in the search endpoint's request body?",
        "Our integration must retry rate-limited requests. Which header gives the delay?",
        "We are migrating to the new SDK and need a replacement for the old upload method.",
        "Can the client library stream a large response without buffering it entirely?",
        "What is the recommended way to create idempotency keys for repeated payment API calls?",
        "How do we subscribe to webhook events for only one project?",
        "Does the batch endpoint preserve the order of the submitted items?",
        "We need an example of authenticating a command-line integration with OAuth.",
        "Which API field distinguishes a soft-deleted record from an active record?",
        "Nothing is down. I am asking how our code should handle a paginated response.",
        "Our users can log in. We need documentation for generating a scoped API credential.",
        "The invoice amount is correct; our code needs help parsing the invoice endpoint schema.",
        "This is a sandbox implementation question, not an incident affecting live service.",
        "No account recovery is needed. How should an application refresh its expiring OAuth token?",
        "Production latency is normal; we want advice on asynchronous SDK calls.",
        "What escaping rules apply when a resource identifier contains a slash?",
        "Can webhook delivery be tested locally using the developer command-line tools?",
        "Please explain the difference between the replacement and patch update methods.",
        "How should our SDK client submit a multipart request containing JSON and a file?",
        "We need to know whether API timestamps are UTC or include an explicit offset.",
        "What response should an integration expect after deleting an already absent resource?",
    ],
}

# These messages were authored separately, not generated by substituting words
# into TRAIN messages. Several explicitly reject an attractive incorrect queue.
DEV = {
    "revenue": [
        "Our auditors require a breakdown of the withholding amount on the latest remittance.",
        "We bought an add-on by mistake. Can its charge be credited without terminating the main plan?",
        "The commercial agreement caps spending, yet this statement exceeds that cap.",
        "A reseller already paid on our behalf; our organization should not also be billed directly.",
        "The technical incident is closed. We are now requesting compensation under the uptime guarantee.",
        "Our integration retrieves receipts successfully. Only the deposit shown on the receipt is disputed.",
        "Please apply the coupon promised by our account representative to the outstanding balance.",
        "We need to reconcile a partial payment with the invoice that still shows the full amount due.",
    ],
    "identity": [
        "The company merged with another business; staff must keep their memberships after their email domain changes.",
        "An administrator accidentally suspended our last remaining owner. We need an authorized recovery procedure.",
        "Our directory group is synchronized, but its members have the wrong access role inside the workspace.",
        "A departed employee is still listed as the person who can approve new organization members.",
        "Payments and page loading are fine. My hardware security key is no longer recognized during authentication.",
        "We are not developing an OAuth client. Please disconnect the suspicious session shown in my personal login history.",
        "An external collaborator was mistakenly given organization-wide privileges instead of one-project access.",
        "Our security team asks whether the compromised user's existing sessions can all be invalidated now.",
    ],
    "reliability": [
        "Database writes are acknowledged as successful, but recently saved records disappear when read back.",
        "The document preview service has returned blank pages for all customers since this morning's deployment.",
        "Our overnight processing backlog grows despite workers reporting themselves healthy.",
        "Users in multiple offices see the same broken attachment downloads, including files that worked yesterday.",
        "The SDK request format is unchanged and valid. A platform regression now rejects it across every customer account.",
        "Sign-in succeeds and permissions are correct, but all report generation crashes with internal server faults.",
        "The customer is not disputing a fee. The payment-status page itself is unavailable across tenants.",
        "A regional scheduler is executing each submitted task twice, even though the requests are unique.",
    ],
    "developer": [
        "We need a code sample that verifies webhook payloads after our reverse proxy decompresses them.",
        "Can an application upload data incrementally while preserving a resumable offset?",
        "Which API version supports conditional updates based on an entity tag?",
        "Our team is designing an integration that distinguishes transient failures from permanent validation errors.",
        "All platform components are healthy. What SDK call cancels an asynchronous export that our application submitted?",
        "Account access is working. We need to implement delegated authorization for a third-party application.",
        "This is not an invoice dispute: which machine-readable field records the tax jurisdiction in the billing API?",
        "Please clarify how the API encodes an empty collection versus an omitted optional field.",
    ],
}

KEYS = list(TRAIN)
CRITERIA = {key: f"The {key.title()} desk." for key in KEYS}
C7 = dict(zip(KEYS, KEYS[1:] + KEYS[:1]))
rows = []
for split, messages in (("train", TRAIN), ("dev", DEV)):
    for topic, texts in messages.items():
        for index, text in enumerate(texts):
            group = f"{split}-{topic}-{index:02d}"
            for contract in ("S1", "C7"):
                shift = index % len(KEYS)
                order = KEYS[shift:] + KEYS[:shift]
                rows.append(dict(id=f"{group}-{contract}", group=group, split=split,
                    state={"company":"Cedar Cloud", "contract_code":contract,
                           "priority":"urgent" if index % 3 == 0 else "routine", "message":text},
                    question={"type":"choice", "instructions":"Apply Cedar Cloud's internal ownership policy and select the final owning desk. Contract codes can override the apparent issue topic. Urgency alone does not change ownership.",
                              "criteria":{key:CRITERIA[key] for key in order}},
                    label=topic if contract == "S1" else C7[topic]))
write_jsonl("support-tickets.jsonl", rows)

# Small, self-authored general replay and disjoint held-out smoke panel.
rows = []
for i in range(72):
    split = "train" if i < 48 else "dev"
    kind = i % 3
    n = 10 + i
    common = dict(id=f"general-{i:03d}", group=f"general-{i:03d}", split=split)
    if kind == 0:
        truth = i % 2 == 0
        rows.append(common | dict(state=f"The supply log lists {n} sealed boxes. All boxes were counted this morning.", question={"type":"noul", "instructions":f"The supply log lists {n if truth else n+1} sealed boxes."}, label=truth))
    elif kind == 1:
        index = (i // 3) % 3
        offers = {"oak":n+3, "pine":n+5, "elm":n+7}
        key = list(offers)[index]
        offers[key] = n
        rows.append(common | dict(state={"quoted_prices":offers, "currency":"USD"}, question={"type":"choice", "instructions":"Choose the supplier with the lowest quoted price.", "criteria":{key:f"Order from {key}." for key in offers}}, label=key))
    else:
        level = (i // 3) % 3
        days = [2, 10, 22][level]
        rows.append(common | dict(state=f"Report {n}: the parcel arrived {days} days after dispatch.", question={"type":"score", "instructions":"Classify the delivery duration using the exact day ranges.", "criteria":["Fast: zero through four days.","Standard: five through fourteen days.","Slow: fifteen days or more."]}, label=level))
write_jsonl("general-decisions.jsonl", rows)
