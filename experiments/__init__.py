from models.model_interface import ModelInterface

PROMPTS = {
    1: {
        "Interleaved": 
            """You are a technical assistant expert in furniture assembly.
            You must determine whether an assembly step was completed, based on the current frame from an assembly process, and according to the instruction manual page.

            Observed frame: <image>
            Instruction manual page: <image>

            Current assembly step number: {step_number}

            Answer "1" if the step was already completed, otherwise "0". You must reply with only "0" or "1".""",
        "Non-Interleaved": 
            """You are a technical assistant expert in furniture assembly.
            You must determine whether an assembly step was completed by comparing the left side of the image (a frame from the assembly process) with the right side (the corresponding instruction manual page).

            Current assembly step number: {step_number}

            Answer "1" if the step was already completed, otherwise "0". You must reply with only "0" or "1"."""
    },
    2: {
        "Interleaved": 
            """You are a technical assistant expert in furniture assembly.
            Your task is to compare a frame from an assembly process with one instruction manual page and determine if the frames show the same step shown in the instruction manual page.

            Observed frame: <image>
            Instruction manual page: <image>

            Answer '1' if the frames correspond to one of the steps in the manual. Answer '0' if they do not.
            You must answer with '1' or '0' only.""",
        "Non-Interleaved": 
            """You are a technical assistant expert in furniture assembly.
            Your task is to compare the left side of the image (a frame from the assembly process) with the right side (the instruction manual page) and determine if the frame matches a step in the manual.

            Answer "1" if the frame corresponds to one of the steps in the manual. Answer "0" if it does not.
            You must answer with "1" or "0" only."""
    },
    3: {
        "Interleaved": 
            """You are a technical assistant expert in furniture assembly.
            Your task is to identify the step number from the instruction manual that matches the action shown in the observed frames.
            You must compare the observed frames to the steps in the instruction manual pages, and pick the correct step.

            Observed frames: <image> <image>
            Instruction manual <image> <image>

            Task: Analyze the observed frames and provide the number of the assembly step being performed.
            You must answer with a single step number.""",
        "Non-Interleaved": 
            """You are a technical assistant expert in furniture assembly.
            Your task is to identify the step number from the instruction manual that matches the action shown in the observed frames.

            The left side of the image contains two frames from the assembly process, and the right side contains two corresponding instruction manual pages.
            Compare the observed frames with the manual steps and determine which step is being performed.

            You must answer with a single step number."""
    }
}

def get_prompt(exp_index, model_instance: ModelInterface):
    return PROMPTS[exp_index]["Interleaved"] if model_instance.supports_interleaved_text_image() else PROMPTS[exp_index]["Non-Interleaved"]