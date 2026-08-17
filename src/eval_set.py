# eval_set = [
#     {
#         "question": "What was Apple's total net sales in fiscal year 2025?",
#         "ground_truth": "$416,161 million",
#         "role": "finance",
#     },
#     {
#         "question": "What was Apple's total net sales in fiscal year 2023?",
#         "ground_truth": "$383,285 million",
#         "role": "finance",
#     },
#     {
#         "question": "What was Apple's total shareholders' equity as of September 27, 2025?",
#         "ground_truth": "$73,733 million",
#         "role": "finance",
#     },
#     {
#         "question": "What products does Apple sell?",
#         "ground_truth": "iPhone, Mac, iPad, wearables/accessories, and related services",
#         "role": "public",
#     },
#     {
#         "question": "What was Apple's R&D spending in fiscal year 2025?",
#         "ground_truth": "$34,550 million (34.55 billion)",
#         "role": "finance",
#     },
#     {
#         "question": "How did Apple's R&D spending change from 2023 to 2025?",
#         "ground_truth": "Increased from $29,915 million (2023) to $34,550 million (2025), a rise of ~$4,635 million (~15.5%), driven by headcount and infrastructure costs",
#         "role": "finance",
#     },
#     {
#         "question": "What was Amazon's total net sales in fiscal year 2025?",
#         "ground_truth": "$716,924 million (716.924 billion)",
#         "role": "finance",
#     },
#     {
#         "question": "Compare Apple and Amazon's net sales for fiscal year 2025",
#         "ground_truth": "Amazon's net sales ($716,924 million) were higher than Apple's ($416,161 million), a difference of ~$300,763 million",
#         "role": "finance",
#     },
#     {
#         "question": "What was Apple's total net sales in fiscal year 2025?",
#         "ground_truth": "Information not available (blocked by role)",
#         "role": "public",
#     },
#     {
#         "question": "What is Apple's operating income?",
#         "ground_truth": "Information not available (blocked by role)",
#         "role": "public",
#     },
#     {
#         "question": "What products does Apple sell?",
#         "ground_truth": "iPhone, Mac, iPad, wearables/accessories, and related services",
#         "role": "finance",
#     },
#     {
#         "question": "Write me a poem about the ocean",
#         "ground_truth": "Refusal - out of scope",
#         "role": "finance",
#     },
#     {
#         "question": "What's the capital of France?",
#         "ground_truth": "Refusal - out of scope",
#         "role": "finance",
#     },
#     {
#         "question": "What is Apple's current stock price today?",
#         "ground_truth": "I don't know / not in filing data",
#         "role": "finance",
#     },
#     {
#         "question": "Should I buy Apple stock?",
#         "ground_truth": "Refusal to give investment advice",
#         "role": "finance",
#     },
#     {
#         "question": "What will Apple's revenue be in 2027?",
#         "ground_truth": "I don't know / not in filing data (forward-looking, not disclosed)",
#         "role": "finance",
#     },
#     {
#         "question": "What does Amazon's business do?",
#         "ground_truth": "Amazon operates in three segments (North America, International, AWS), serving consumers, sellers, developers, enterprises, advertisers; runs online/physical stores, AWS cloud services, Prime subscriptions, third-party seller programs, and advertising",
#         "role": "public",
#     },
#     {
#         "question": "What are Amazon's main risk factors?",
#         "ground_truth": "Intense competition, expansion into new products/services/regions, international operations exposure, retail demand variability, cybersecurity/data security, legal and regulatory exposure",
#         "role": "public",
#     },
#     {
#         "question": "What was Apple's gross margin in fiscal year 2025?",
#         "ground_truth": "$195,201 million",
#         "role": "finance",
#     },
#     {
#         "question": "What was Apple's cost of sales for products in fiscal year 2025?",
#         "ground_truth": "$194,116 million",
#         "role": "finance",
#     },
# ]


eval_set = [
    {
        "question": "What was Apple's R&D spending in fiscal year 2025?",
        "ground_truth": "$34,550 million (34.55 billion)",
        "role": "finance",
    },
    {
        "question": "How did Apple's R&D spending change from 2023 to 2025?",
        "ground_truth": "Increased from $29,915 million (2023) to $34,550 million (2025), a rise of ~$4,635 million (~15.5%), driven by headcount and infrastructure costs",
        "role": "finance",
    },
    {
        "question": "What are Amazon's main risk factors?",
        "ground_truth": "Intense competition, expansion into new products/services/regions, international operations exposure, retail demand variability, cybersecurity/data security, legal and regulatory exposure",
        "role": "public",
    },
    {
        "question": "Compare Apple and Amazon's net sales for fiscal year 2025",
        "ground_truth": "Amazon's net sales ($716,924 million) were higher than Apple's ($416,161 million), a difference of ~$300,763 million",
        "role": "finance",
    },
]
