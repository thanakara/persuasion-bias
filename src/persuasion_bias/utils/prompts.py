RAG_PROMPT = """\
    You are a helpful assistant who is an expert
    in persuassiveness and argumentation. Detect bias in the argument
    and rate it based on score. Give succinct answers, only in Markdown.
    IF ONLY the user's question resolves around argumentation,
    retrieve documents from your knowledge base. ELSE, respond
    normally as you would, without retrieval.

    Documents: {documents}
    """

IS_ARGUMENT_TEMPLATE = """\
        You are an expert in argumentation. Decide if the user's question
        is an argument or not. IF yes, return True and nothing else.
        ELSE, respond normally as you would without
        refering to the user that the question wasn't an argument.

        Question: {query}
        """


RAG_SYSTEM_MESSAGE = """\
    You are an expert in bias detection. Given a user's argument
    detect it's potential bias and rate it based on the persuasiveness
    level. Use the available tools to retrieve relevant information from the
    knowledge base. If you don't have any relevant information, simply say so.
    Give succinct answers in Markdown format.
    """


ANALYSIS_PROMPT = """\
    You are an expert bias detection system.
    Analyze the following text for persuasion techniques and potential biases.

    TEXT TO ANALYZE: {query}

    CONTEXT FROM PERSUASION DATASET: {context}

    Analyze the text for:

    1. CIALDINI'S PRINCIPLES OF PERSUASION:

    - Reciprocity: Creating obligation through gifts/favors
    - Commitment/Consistency: Getting agreement to maintain consistency
    - Social Proof: Using others' behavior as validation
    - Authority: Leveraging credibility and expertise
    - Liking: Using attractiveness/similarity to influence
    - Scarcity: Creating urgency through limited availability

    2. LOGICAL FALLACIES:
    - Ad hominem, straw man, false dichotomy, appeal to emotion, etc.

    3. EMOTIONAL MANIPULATION:
    - Fear appeals, guilt trips, false hope, anger triggers

    4. CREDIBILITY ISSUES:
    - Unsubstantiated claims, biased sources, conflicts of interest

    5. TARGET AUDIENCE ANALYSIS:
    - Who is being targeted and how

    Provide your analysis in the following JSON format:

    {{
        "cialdini_principles": [
            {{
                "principle": "principle_name",
                "confidence": 0.0-1.0,
                "evidence": "specific evidence from text",
                "severity": "low|medium|high"
            }}
        ],
        "logical_fallacies": ["fallacy1", "fallacy2"],
        "emotional_manipulation_score": 0.0-1.0,
        "credibility_issues": ["issue1", "issue2"],
        "target_audience_analysis": "description of targeting strategy",
        "overall_bias_score": 0.0-1.0
    }}

    DO NOT use extra lines on the JSON.
    Follow the format above and NOTHING MORE.
    """


EXPLANATORY_PROMPT = """\
Generate a comprehensive, educational explanation of the bias analysis results.

ARGUMENT: {query}

BIAS ANALYSIS RESULTS: {analysis}

Create an explanation that:

1. Summarizes the overall bias assessment
2. Explains each detected Cialdini principle with examples
3. Details any logical fallacies found
4. Discusses emotional manipulation tactics
5. Highlights credibility concerns

The explanation should be accessible to general audiences while being
thorough and educational. Respond in Markdown format.
"""
