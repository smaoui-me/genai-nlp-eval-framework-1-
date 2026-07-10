"""Ready-made example texts, so the Annotate page has something to try
immediately without needing to upload a file first.

SAMPLES is a dict: each key is the name shown in the dropdown, each value
is the text that gets annotated when that sample is picked.
"""

SAMPLES = {
    "Support ticket — delayed delivery": (
        "Customer reported a delayed delivery in Munich after shipment from DHL Hub. "
        "Please contact John Miller before Friday and verify the destination address in Berlin. "
        "The original order was placed through Amazon Logistics and the package left the "
        "Frankfurt warehouse on Tuesday."
    ),
    "News snippet — product launch": (
        "Apple unveiled the new iPhone at their Cupertino headquarters during WWDC, with CEO "
        "Tim Cook presenting the keynote. The company also announced a partnership with "
        "Foxconn to expand production in Shenzhen ahead of the holiday season."
    ),
    "Support ticket — billing dispute": (
        "Maria Gonzalez contacted support about a duplicate charge from her subscription with "
        "Northwind Traders. The invoice was issued in Chicago on March 3rd, and she asked to "
        "be transferred to the billing team lead, David Chen, for a refund."
    ),
}
