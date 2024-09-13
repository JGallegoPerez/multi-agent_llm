import os
import re
import base64
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv, find_dotenv
import anthropic
from openai import OpenAI
from ultralytics import YOLOv10
import cv2
import supervision as sv
from supervision import Position
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
from langchain_community.llms import Ollama
from typing_extensions import TypedDict
from typing import Annotated, Literal
from langgraph.graph.message import add_messages
import json
import pygame
from gtts import gTTS
from pydub import AudioSegment
from PIL import Image, ImageDraw
from IPython.display import display

# Environment and API key functions

def load_env():
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())

def get_api_key(key_name: str) -> str:
    """Retrieve API key from environment variables."""
    load_env()
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"{key_name} not found in environment variables.")
    return api_key

def get_openai_api_key() -> str:
    """Get OpenAI API key."""
    return get_api_key("OPENAI_API_KEY")

def get_anthropic_api_key() -> str:
    """Get Anthropic API key."""
    return get_api_key("ANTHROPIC_API_KEY")

def get_tavily_api_key() -> str:
    """Get Tavily API key."""
    return get_api_key("TAVILY_API_KEY")

# Chatbot and agent creation functions

def create_chatbot(llm, tools, system_prompt):
    """Create an agent with the given LLM, tools, and system prompt."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | llm.bind_tools(tools)

def chatbot_node(state, agent, name):
    """Create a node for a given agent in the LangGraph."""
    print("> inside CHATBOT node")
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        return {"messages": [result], "sender": name}
    return {
        "messages": [AIMessage(**result.dict(exclude={"type", "name"}), name=name)],
        "sender": name,
    }

# Tool node and routing

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: List[Any]) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, List[ToolMessage]]:
        print("> inside BasicToolNode")
        if not (messages := inputs.get("messages", [])):
            raise ValueError("No message found in input")
        
        message = messages[-1]
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

def route_tools(state: State) -> Literal["tools", "__end__"]:
    """Route to the ToolNode if the last message has tool calls. Otherwise, route to the end."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    return "tools" if hasattr(ai_message, "tool_calls") and ai_message.tool_calls else "__end__"

# Text-to-speech function

def text_to_speech(text: str, playback_speed: float = 1.3) -> None:
    """Convert text to speech and play it."""
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")

    audio = AudioSegment.from_mp3("response.mp3")
    faster_audio = audio.speedup(playback_speed=playback_speed)
    faster_audio.export("faster_response.mp3", format="mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("faster_response.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

    os.remove("response.mp3")
    os.remove("faster_response.mp3")

# YOLO and image processing functions

def yolo_for_llm_lst(image_path: str, show_image: bool = False) -> List[Dict[str, Any]]:
    """Run YOLO object detection on an image and return results as a list of dictionaries."""
    model_path = os.path.join(os.getcwd(), 'yolo/weights/best.pt')
    model = YOLOv10(model_path)

    img = cv2.imread(image_path)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=1)
    label_annotator = sv.LabelAnnotator(text_scale=0.2, text_padding=2, text_position=Position.BOTTOM_LEFT)

    results = model(source=img, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence 
        in zip(detections['class_name'], detections.confidence)
    ]

    annotated_image = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    if show_image:
        sv.plot_image(annotated_image)

    return [
        {
            "class": f"{class_name} block" if class_name != "gripper" else class_name,
            "x_min": round(xyxy[0].item()),
            "y_min": round(xyxy[1].item()),
            "x_max": round(xyxy[2].item()),
            "y_max": round(xyxy[3].item())
        }
        for xyxy, class_name in zip(detections.xyxy, detections.data['class_name'])
    ]

# LLM interpretation and summarization functions

def llm_interpret_yolo(yolo_info: str, instructions: str = None) -> str:
    """Interpret YOLO results using an LLM."""
    MODEL = "gpt-4o"
    openai_client = OpenAI(api_key=get_openai_api_key())

    if instructions is None:
        prompt = f"""
        The following YOLO data corresponds to bounding boxes of colored blocks (absence of blocks is possible).
        The blocks are on a table surface, although some blocks can be on top of each other.
        Higher values for the x coordinate indicate points are more to the right.
        Higher values for the y coordinate indicate points are lower.
        LOWER values for the y coordinate indicate points are HIGHER.
        {yolo_info}
        Analyze the blocks configuration first by splitting it in columns ('columns are vertical structures').
        ONE BLOCK CAN HAVE ONLY ONE BLOCK ON TOP (if any) AND BE UNDER ONE BLOCK (if any). You might find horizontal overlappings.
        Force the column-grouping by what is most likely. WHEN THE y_max OF TWO BLOCKS IS APPROXIMATELY THE SAME, IT MEANS THAT THEY ARE ON THE SAME ROW.
        Then, analyze the configuration analogously, but in terms of rows ('rows are horizontal structures'). State explicitly the order of the blocks IN EACH ROW,
        from left to right.
        If the bounding boxes of two blocks overlap at least a little, it means the two blocks are in contact with each other.
        Remember: Higher values for the y coordinate indicate points are lower. LOWER values for the y coordinate indicate points are HIGHER.
        If the y-coordinates of two blocks are SIMILAR, it means they are on the same plane; if the x-coordinates of two blocks are SIMILAR, they are on the same column. Don't use words like 'slightly' or 'a little'.
        At the conclusion, DON'T SPEAK IN TERMS OF COLUMNS AND ROWS, just mention what is left of what, what right of what, what on top of what, etc.) If two bounding boxes don't overlap AT ALL, the corresponding blocks are not next to each other.
        ONLY ONCE YOU REACHED A CONCLUSION: do the blocks appear to have a specific overall shape?
        """
    else:
        prompt = f"""
        The following YOLO data corresponds to several objects, which might include colored blocks and a robot gripper. (Or maybe nothing):
        {yolo_info}
        Higher values for the x coordinate indicate that points are more to the right.
        Higher values for the y coordinate indicate that points are lower.
        If there is a color in a block's class, it is because the block it represents has that color.
        We want to target one specific block from the YOLO data, according to the following instruction:
        {instructions}
        Return the YOLO data for the only block that meets the instruction, excluding all other objects.
        Provide your answer in YOLO format, with the class name and the bounding box coordinates.
        The answer should therefore look like {{'class': ... block', 'x_min': ..., 'y_min': ..., 'x_max': ..., 'y_max': ...}}.
        DON'T ADD ANYTHING ELSE TO THE ANSWER.
        """

    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You interpret YOLO results from bounding boxes coordinates."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return completion.choices[0].message.content

def llm_summarize(llm_yolo_output: str) -> str:
    """Summarize the LLM's interpretation of YOLO results."""
    MODEL = "gpt-4o-mini"
    openai_client = OpenAI(api_key=get_openai_api_key())
    prompt = f"""
    Look at the conclusions and summarize the spatial relationships between ALL THE BLOCKS the objects in the following text.
    Don't add any numeric information; don't add text other than the plain summary:
    {llm_yolo_output}
    """

    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Summarize, particularly at the conclusions."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

def lmm_no_blocks(image_path: str) -> str:
    """Use a vision model to describe non-block objects in the image."""
    lmm_prompt_excluding = """
    There is a white foam table with several objects (or nothing) on top of it or above it.
    Mention in ONE SENTENCE what you see in the image, EXCLUDING the table;
    EXCLUDING a grey metallic, cylindrical robot part in the background;
    EXCLUDING any robotic gripper if you see one;
    DON'T MENTION ANYTHING RELATED TO COLORED BLOCKS!.
    If you don't see anything else, say 'Nothing else to add about the image'
    Otherwise, mention the remaining things you see, especially if an action is happening.
    For example, the actions could consist in hands grasping specific colored blocks or other objects.
    """

    MODEL = "gpt-4o-mini"
    openai_client = OpenAI(api_key=get_openai_api_key())

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(image_path)

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": lmm_prompt_excluding},
            {"role": "user", "content": [
                {"type": "text", "text": "What do you see?"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def llm_combine_all(yolo_summary: str, not_blocks: str) -> str:
    """Combine summaries of blocks and non-block objects."""
    MODEL = "gpt-4o-mini"
    openai_client = OpenAI(api_key=get_openai_api_key())
    prompt = f"""
    There is an image with colored blocks (or none) and other potential objects (or none), on/around a table.
    Merge the two following incomplete descriptions of the image into a single description.
    Avoid repeating the same objects.
    DON'T ADD any other text, JUST MERGE THE DESCRIPTIONS:
    Description about colored blocks (maybe no blocks): {yolo_summary}
    Description about other things in the image (maybe nothing): {not_blocks}
    """
    completion = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Merge the two descriptions."},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

# Image manipulation function

def arrow_point(image_path: str, tip_x: int, tip_y: int, arrow_color: Tuple[int, int, int] = (255, 0, 0), arrow_size: int = 60) -> None:
    """Insert a down-pointing arrow at specified coordinates in the image and display it."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    left_x = tip_x - arrow_size // 2
    right_x = tip_x + arrow_size // 2
    bottom_y = tip_y + arrow_size

    tip_y -= 40
    bottom_y -= 40

    draw.polygon([(left_x, tip_y), (right_x, tip_y), (tip_x, bottom_y)], fill=arrow_color)
    display(image)
