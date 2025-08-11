# Prompt templates
from enum import Enum
from typing import Dict,Tuple

class ExplanationLevel(str, Enum):
    CHILD = "child"
    TEENAGER = "teenager"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    EXPERT = "expert"

# Prompt templates for different explanation levels
LEVEL_PROMPTS: Dict[ExplanationLevel, Dict[str,str]] = {
    ExplanationLevel.CHILD: {
        "system": """you are explaining to a 5-years old child. use 
        - very simple words (no big words )
        - short sentences
        - fun analogies from everyday life (toys, animals, cartoons)
        - no technical terms
        - encouraging and positive tone
        - use "imagine if .. " or "think about .." to start the explanation
        """,
        "template": """explain this topic like am 5 years old {topic}
        here some instructions that might help: {context}:
        remember to :
        - Use only simple words that a child would understand
        - make it fun and intersting 
        - use examples from things a child knows (toys, animals, cartoons)
        - ask simple questions to engage the child  
        """,
    },
    ExplanationLevel.TEENAGER: {
        "system": """you are explaining to a curious 15 years old teenager. use 
        - Clear, engaging language they can understand
        - Examples from technology, social media, sports, movies
        - Some technical terms but explain them simply
        - Relatable analogies from their world
        - Show why it's relevant to their life""",
        "template": """Explain this topic for a teenager: {topic}
        Context information: {context}
        Make it:
        - Interesting and relevant to teenage life
        - Clear but not oversimplified
        - Include why they should care about this
        - Use examples they can relate to
        - Connect to things they already know about"""
    },
    ExplanationLevel.UNDERGRADUATE: {
        "system": """You are explaining to a university student. Use:
        - Academic but accessible language
        - Proper terminology with clear definitions
        - Structured explanations with key concepts
        - Examples from various fields of study
        - Show connections to broader knowledge""",
                
        "template": """Provide an undergraduate-level explanation of: {topic}
        Context from source material: {context}

        Include:
        - Key concepts and important terminology
        - How this connects to other subjects
        - Real-world applications and examples
        - Clear logical structure
        - Why this knowledge is important"""
    },
            
    ExplanationLevel.GRADUATE: {
        "system": """You are explaining to a graduate student. Use:
        - Advanced academic language
        - Technical precision and depth
        - Critical analysis and evaluation
        - References to current research and methods
        - Nuanced understanding of complexities""",
        "template": """Provide a graduate-level analysis of: {topic}

        Source context: {context}

        Focus on:
        - Technical depth and precision
        - Current research and developments
        - Critical analysis and implications
        - Advanced methodologies and theories
        - Connections to cutting-edge work in the field"""
    },
            
    ExplanationLevel.EXPERT: {
        "system": """You are communicating with a domain expert. Use:
        - Highly technical and precise language
        - Advanced concepts without extensive explanation
        - Latest research findings and ongoing debates
        - Nuanced analysis and cutting-edge perspectives
        - Assumption of deep background knowledge""",
        "template": """Provide an expert-level analysis of: {topic}

        Context material: {context}

        Include:
        - Latest research developments and findings
        - Technical nuances and edge cases
        - Current debates and open questions
        - Advanced theoretical frameworks
        - Implications for future work"""
    }
}
def get_prompt_for_level(level:ExplanationLevel,topic:str,context:str) -> Tuple[str,str]:
    if level not in LEVEL_PROMPTS:
        raise ValueError(f"Invalid explanation level: {level}")
    
    prompt_info = LEVEL_PROMPTS[level]
    system_message = prompt_info["system"]
    user_prompt = prompt_info["template"].format(topic=topic, context=context)

    complete_prompt = f"""System instruction : {system_message}
    user request : {user_prompt}
    provide a detailed and accurate response based on the provided context.
    """

    return complete_prompt
def get_available_levels() -> Dict[str, str]:
    """Get available explanation levels with descriptions"""
    return {
        ExplanationLevel.CHILD.value: "Child (5 years old) - Simple words and fun examples",
        ExplanationLevel.TEENAGER.value: "Teenager (15 years old) - Engaging and relatable", 
        ExplanationLevel.UNDERGRADUATE.value: "University Student - Academic but accessible",
        ExplanationLevel.GRADUATE.value: "Graduate Student - Advanced and technical",
        ExplanationLevel.EXPERT.value: "Expert - Highly technical and precise"
    }