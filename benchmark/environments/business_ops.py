"""Tier 3 — business operations (CRM, billing, orders, outbound mail).

The hard tier, and hard in a specific, measurable way rather than by vibe:

* twenty tools, four times the size at which Needle switches from rendering
  every schema to retrieving a subset;
* six tools whose names begin `search_`, five that begin `send_`, three that
  begin `get_` — the verb carries almost no routing signal, so the object has
  to;
* several entity types that legitimately co-occur in one sentence ("John's
  latest order", "the invoice for Northwind"), so a distractor is not a
  contrived trap but the normal shape of the request;
* destructive operations (`delete_customer`, `issue_refund`) sitting next to
  benign lookups, which makes a false action expensive rather than merely wrong.

This tier is where the spec's original question — does performance fall when
tool descriptions become semantically close — actually gets tested.
"""
from __future__ import annotations

from ._schema import apply_overrides, param, tool

ENV_ID = "business_ops"
TIER = 3

READ_ONLY = {"search_customer", "search_company", "search_contact", "search_invoice",
             "search_order", "search_product", "get_customer", "get_invoice", "get_order"}


def initial_state(overrides: dict | None = None) -> dict:
    return apply_overrides({
        "customers": {
            "cus_1": {"name": "John Marsh", "email": "john.marsh@example.com",
                      "company": "Northwind Ltd", "deleted": False, "notes": ""},
            "cus_2": {"name": "Aisha Khan", "email": "aisha.khan@example.com",
                      "company": "Beacon Co", "deleted": False, "notes": ""},
        },
        "companies": {"com_1": {"name": "Northwind Ltd", "tier": "enterprise"},
                      "com_2": {"name": "Beacon Co", "tier": "smb"}},
        "contacts": {"con_1": {"name": "Ravi Desai", "company": "Northwind Ltd",
                               "email": "ravi.desai@example.com"}},
        "invoices": {
            "inv_1": {"customer": "cus_1", "amount": 4200, "status": "unpaid",
                      "date": "2026-08-30"},
            "inv_2": {"customer": "cus_2", "amount": 900, "status": "paid",
                      "date": "2026-09-05"},
        },
        "orders": {
            "ord_1": {"customer": "cus_1", "date": "2026-09-12", "status": "shipped",
                      "total": 1500},
            "ord_2": {"customer": "cus_1", "date": "2026-09-18", "status": "pending",
                      "total": 800},
            "ord_3": {"customer": "cus_2", "date": "2026-09-02", "status": "shipped",
                      "total": 300},
        },
        "products": {"prd_1": {"name": "Widget A", "price": 150},
                     "prd_2": {"name": "Widget B", "price": 220}},
        "sent_email": [],
        "refunds": [],
    }, overrides)


# --------------------------------------------------------------------------
# Executors
# --------------------------------------------------------------------------

def _search(bucket, field="name"):
    def run(s, query):
        q = str(query).lower()
        hits = [{"id": k, **v} for k, v in s[bucket].items()
                if q in str(v.get(field, "")).lower() and not v.get("deleted")]
        return {"query": query, "count": len(hits), "results": hits}
    return run


def _search_invoice(s, query):
    q = str(query).lower()
    hits = []
    for k, v in s["invoices"].items():
        cust = s["customers"].get(v["customer"], {})
        if q in k.lower() or q in cust.get("name", "").lower() or q in v["status"]:
            hits.append({"id": k, **v, "customer_name": cust.get("name")})
    return {"query": query, "count": len(hits), "results": hits}


def _search_order(s, query):
    q = str(query).lower()
    hits = []
    for k, v in s["orders"].items():
        cust = s["customers"].get(v["customer"], {})
        if q in k.lower() or q in cust.get("name", "").lower() or q in v["status"]:
            hits.append({"id": k, **v, "customer_name": cust.get("name")})
    hits.sort(key=lambda r: r["date"], reverse=True)
    return {"query": query, "count": len(hits), "results": hits}


def _get(bucket):
    def run(s, **kwargs):
        key = next(iter(kwargs.values()))
        item = s[bucket].get(key)
        return {"id": key, **item} if item else {"error": f"not found: {key}"}
    return run


def _send(kind):
    def run(s, **kwargs):
        record = {"kind": kind, **kwargs}
        s["sent_email"].append(record)
        return {"ok": True, **record}
    return run


def _create_order(s, customer_id, product_id, quantity):
    if customer_id not in s["customers"]:
        return {"error": f"no such customer: {customer_id}"}
    if product_id not in s["products"]:
        return {"error": f"no such product: {product_id}"}
    new_id = f"ord_{len(s['orders']) + 1}"
    s["orders"][new_id] = {"customer": customer_id, "date": "2026-09-20",
                           "status": "pending",
                           "total": s["products"][product_id]["price"] * int(quantity)}
    return {"ok": True, "order_id": new_id}


def _cancel_order(s, order_id):
    order = s["orders"].get(order_id)
    if not order:
        return {"error": f"no such order: {order_id}"}
    order["status"] = "cancelled"
    return {"ok": True, "order_id": order_id, "status": "cancelled"}


def _update_customer(s, customer_id, notes):
    cus = s["customers"].get(customer_id)
    if not cus:
        return {"error": f"no such customer: {customer_id}"}
    cus["notes"] = notes
    return {"ok": True, "customer_id": customer_id}


def _delete_customer(s, customer_id):
    cus = s["customers"].get(customer_id)
    if not cus:
        return {"error": f"no such customer: {customer_id}"}
    cus["deleted"] = True
    return {"ok": True, "customer_id": customer_id}


def _mark_invoice_paid(s, invoice_id):
    inv = s["invoices"].get(invoice_id)
    if not inv:
        return {"error": f"no such invoice: {invoice_id}"}
    inv["status"] = "paid"
    return {"ok": True, "invoice_id": invoice_id}


def _issue_refund(s, order_id, amount):
    if order_id not in s["orders"]:
        return {"error": f"no such order: {order_id}"}
    s["refunds"].append({"order_id": order_id, "amount": int(amount)})
    return {"ok": True, "order_id": order_id, "amount": int(amount)}


EXEC = {
    "search_customer": _search("customers"), "search_company": _search("companies"),
    "search_contact": _search("contacts"), "search_product": _search("products"),
    "search_invoice": _search_invoice, "search_order": _search_order,
    "get_customer": _get("customers"), "get_invoice": _get("invoices"),
    "get_order": _get("orders"),
    "send_email": _send("general"), "send_invoice_email": _send("invoice"),
    "send_marketing_email": _send("marketing"), "send_internal_email": _send("internal"),
    "send_followup_email": _send("followup"),
    "create_order": _create_order, "cancel_order": _cancel_order,
    "update_customer": _update_customer, "delete_customer": _delete_customer,
    "mark_invoice_paid": _mark_invoice_paid, "issue_refund": _issue_refund,
}


def executors(profile) -> dict:
    return EXEC


# --------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------

# Trigger regexes, declared by tool name rather than threaded through each
# call site. A matching trigger restricts decoding to the matched tools and
# forces a call, so the table is deliberately conservative: a trigger that
# fires on an under-specified request converts a correct refusal into a wrong
# call, which is exactly what the opt_triggers arm is there to measure.
TRIGGERS = {
    'search_customer': [r'\\b(customers?|who is)\\b'],
    'search_company': [r'\\b(compan(y|ies)|organisation|organization)\\b'],
    'search_contact': [r'\\bcontacts?\\b'],
    'search_invoice': [r'\\b(invoices?|bills?|owes?|owed|unpaid)\\b'],
    'search_order': [r'\\b(orders?|bought|purchase[sd]?)\\b'],
    'search_product': [r'\\b(products?|widget|catalogue|catalog)\\b'],
    'cancel_order': [r'\\b(cancel|kill)\\b.{0,25}\\border\\b'],
    'mark_invoice_paid': [r'\\b(settle|mark .{0,20}paid)\\b'],
    'issue_refund': [r'\\brefund\\b'],
    'delete_customer': [r'\\bdelete\\b.{0,25}\\bcustomer\\b'],
}


def _with_triggers(schemas, profile):
    if not profile.triggers:
        return schemas
    for schema in schemas:
        pats = TRIGGERS.get(schema["name"])
        if pats:
            schema["triggers"] = list(pats)
    return schemas


def tools(p) -> list[dict]:
    def q(naive, tuned):
        return {"query": param(p, "string", naive, tuned)}

    cus_id = param(p, "string", "Customer id.",
                   "The customer identifier, 'cus_N'. Obtain it from search_customer; "
                   "never invent one.", pattern=r"^cus_\d+$")
    inv_id = param(p, "string", "Invoice id.",
                   "The invoice identifier, 'inv_N'. Obtain it from search_invoice.",
                   pattern=r"^inv_\d+$")
    ord_id = param(p, "string", "Order id.",
                   "The order identifier, 'ord_N'. Obtain it from search_order.",
                   pattern=r"^ord_\d+$")

    def mail(name, naive, tuned):
        return tool(p, name, naive, tuned,
                    {"to": param(p, "string", "Recipient.",
                                 "The recipient's email address or name, as given."),
                     "subject": param(p, "string", "Subject line.",
                                      "The subject line, copied from the request."),
                     "body": param(p, "string", "Message body.",
                                   "The message text, copied from the request.")},
                    ["to", "subject", "body"])

    return _with_triggers([
        # --- lookups: six tools, one verb, six different objects -----------
        tool(p, "search_customer", "Search for a customer.",
             "Find a person who buys from us, by their name. Returns customer records "
             "only; not companies, not contacts.",
             q("Name to search.", "The customer's name as written in the request."),
             ["query"]),
        tool(p, "search_company", "Search for a company.",
             "Find an organisation by its name. Returns company records only; a named "
             "individual is a customer or a contact, not a company.",
             q("Name to search.", "The company name as written in the request."), ["query"]),
        tool(p, "search_contact", "Search for a contact.",
             "Find a named person at a company who is not themselves a buyer. Returns "
             "contact records only.",
             q("Name to search.", "The contact's name as written in the request."), ["query"]),
        tool(p, "search_invoice", "Search invoices.",
             "Find bills we have issued, by customer name or status. Use for anything "
             "about money owed, bills, or payment. Not for orders.",
             q("Search text.", "A customer name, or a status such as paid or unpaid."),
             ["query"]),
        tool(p, "search_order", "Search orders.",
             "Find purchases a customer has placed, by customer name or status, newest "
             "first. Use for anything about what someone bought or ordered. Not for "
             "invoices.",
             q("Search text.", "A customer name, or a status such as pending or shipped."),
             ["query"]),
        tool(p, "search_product", "Search the product catalogue.",
             "Find an item we sell, by product name. Returns catalogue entries, not orders.",
             q("Product name.", "The product name as written in the request."), ["query"]),

        # --- detail reads ---------------------------------------------------
        tool(p, "get_customer", "Get a customer record.",
             "Read one customer's full record by id.", {"customer_id": cus_id},
             ["customer_id"]),
        tool(p, "get_invoice", "Get an invoice.",
             "Read one invoice by id.", {"invoice_id": inv_id}, ["invoice_id"]),
        tool(p, "get_order", "Get an order.",
             "Read one order by id.", {"order_id": ord_id}, ["order_id"]),

        # --- outbound mail: five tools, one verb, differing only in purpose --
        mail("send_email", "Send an email.",
             "Send an ordinary one-off email. Use only when none of the more specific "
             "mail tools applies."),
        mail("send_invoice_email", "Send an invoice email.",
             "Email a customer about a bill they owe. Use when the message is about an "
             "invoice or a payment."),
        mail("send_marketing_email", "Send a marketing email.",
             "Email a promotion, offer, or campaign to a customer. Use only when the "
             "request is explicitly promotional."),
        mail("send_internal_email", "Send an internal email.",
             "Email a colleague inside our own company. Use when the recipient is a "
             "teammate rather than a customer."),
        mail("send_followup_email", "Send a follow-up email.",
             "Email a customer to chase an earlier conversation. Use when the request "
             "says to follow up or chase."),

        # --- mutations -------------------------------------------------------
        tool(p, "create_order", "Create an order.",
             "Place a new order for a customer. Every field must come from the request.",
             {"customer_id": cus_id,
              "product_id": param(p, "string", "Product id.",
                                  "The product identifier, 'prd_N', from search_product.",
                                  pattern=r"^prd_\d+$"),
              "quantity": param(p, "integer", "How many.",
                                "How many units, as a whole number from the request.",
                                bounds=(1, 1000))},
             ["customer_id", "product_id", "quantity"]),
        tool(p, "cancel_order", "Cancel an order.",
             "Cancel a placed order by id. Only on an explicit instruction to cancel.",
             {"order_id": ord_id}, ["order_id"]),
        tool(p, "update_customer", "Update a customer record.",
             "Attach a note to a customer's record.",
             {"customer_id": cus_id,
              "notes": param(p, "string", "The note.",
                             "The note text, copied from the request.")},
             ["customer_id", "notes"]),
        tool(p, "delete_customer", "Delete a customer.",
             "Permanently remove a customer record. Destructive and irreversible. Only "
             "on an explicit, unambiguous instruction to delete the customer.",
             {"customer_id": cus_id}, ["customer_id"]),
        tool(p, "mark_invoice_paid", "Mark an invoice as paid.",
             "Record that an invoice has been settled.",
             {"invoice_id": inv_id}, ["invoice_id"]),
        tool(p, "issue_refund", "Issue a refund.",
             "Return money for an order. Destructive and irreversible. The amount must "
             "be stated in the request.",
             {"order_id": ord_id,
              "amount": param(p, "integer", "Amount.",
                              "The refund amount in whole currency units, from the "
                              "request.", bounds=(1, 100000))},
             ["order_id", "amount"]),
    ], p)


def translate_gold(calls: list[dict], profile) -> list[dict]:
    return [dict(c) for c in calls]
