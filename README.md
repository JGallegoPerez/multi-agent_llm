# multi-agent_llm
## A multi-agent, multimodal system that can incorporates RAG, memory and planning; and can take input from other traditional ML algorithms (e.g. CNNs).

The present repository is based on previous and ongoing work at the Okinawa Institute of Science and Technology (OIST). A multi-agent system was developed, which coordinated several LLMs, a fine-tuned Convolutional Neural Network for image detection, and a robotic arm. The system was designed to simulate a teaching environment, where an AI agent took on the role of a mother guiding a child in manipulating colored cubes. It was capable of real-time communication in open natural language (speech), object detection, reasoning, memory, and improvisation. 

The system's architecture leveraged the use of several agents within three interconnected process columns, namely Vision, Reasoning (chatbot) and Robotic Actions:

![multi_agent_architecture](https://github.com/user-attachments/assets/05b2dc4a-32dd-4592-ab15-cbcf22691888)


The following video comes from a longer human-robot interaction. It illustrates the open-language, reasoning capabilties of the system. Notice how the system succesfully accomplishes a task based on indirect instructions (to point with the robotic arm at a block that is the "color of the sky").

https://github.com/user-attachments/assets/aafd5a66-2225-416d-8d04-1f99cc9c2c8e


## Project Adaptation

This repository contains an adapted version of the original multi-agent system, designed to demonstrate its core functionalities without the need for specialized hardware. Here's an overview of the adaptation and how to use it:

### Contents

- *task_images_tiny*: Folder containing sample images for the system to process
- *README.md*: This file, providing project overview and instructions
- *agents_JorgeGallego.pdf*: Presentation showcasing the adapted system at AI Innovators talk
- *blocks_agentic.ipynb*: Jupyter notebook containing the main implementation
- *requirements.txt*: List of Python dependencies
- *utils.py*: Utility functions used by the main system
- *.env*: File for storing API credentials (not included in repository)
- *voice_recog.py*: Module for speech recognition functionality

### Setup

1. Clone this repository:
   
   git clone https://github.com/JGallegoPerez/multi-agent_llm.git
   
   cd multi-agent_llm
   

3. Install the required dependencies:
   
   pip install -r requirements.txt
   

4. Set up your *.env* file with the necessary API credentials:
   
   OPENAI_API_KEY=your_openai_api_key
   
   ANTHROPIC_API_KEY=your_anthropic_api_key
   
   TAVILY_API_KEY=your_tavily_api_key
   

### Usage

1. Open the *blocks_agentic.ipynb* notebook in Jupyter Lab or Jupyter Notebook.

2. Follow the instructions in the notebook to run the multi-agent system.

3. The system will process images from the *task_images_tiny* folder, simulating real-time camera input.

4. Interact with the system using text input or voice commands (if speech recognition is enabled).

### Key Features

- **Multimodal Processing**: Combines image analysis (YOLO) with natural language understanding (LLMs).
- **Multi-Agent Coordination**: Utilizes multiple AI models for different tasks (e.g., vision, dialogue, reasoning).
- **RAG (Retrieval-Augmented Generation)**: Incorporates external knowledge when needed.
- **Memory and Context Management**: Maintains conversation history and context across interactions.
- **Speech Recognition**: Optional voice input for more natural interaction (see *voice_recog.py*).

### Limitations

- This adaptation does not include the robotic arm functionality of the original system.
- The image processing is simulated using pre-captured images rather than real-time camera input.

### Future Improvements

- Integration with real-time camera input
- Expansion of the knowledge base and reasoning capabilities
- Implementation of more advanced planning algorithms

For any questions or issues, please open an issue in this repository.




