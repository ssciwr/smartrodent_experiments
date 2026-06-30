import ammico
import os

def load_images(data_path):
    # Read your data into AMMICO
    # `ammico` reads in files from a directory. You can iterate through directories in a recursive manner and filter by extensions. Note that the order of the files may vary on different OS. Reading in these files creates a dictionary `image_dict`, with one entry per image file, containing the file path and filename. This dictionary is the main data structure that ammico operates on and is extended successively with each detector run as explained below.
    # For reading in the files, the ammico function `find_files` is used, with optional keywords:
    # | input key | input type | possible input values |
    # | --------- | ---------- | --------------------- |
    # `path` | `str` | the directory containing the image files (defaults to the location set by environment variable `AMMICO_DATA_HOME`) |
    # | `pattern` | `str\|list` | the file extensions to consider (defaults to "png", "jpg", "jpeg", "gif", "webp", "avif", "tiff") |
    # | `recursive` | `bool` | include subdirectories recursively (defaults to `True`) |
    # | `limit` | `int` | maximum number of files to read (defaults to `20`, for all images set to `None` or `-1`) |
    # | `random_seed` | `int` | the random seed for shuffling the images; applies when only a few images are read and the selection should be preserved (defaults to `None`) |
    # Find files and create the image dictionary
    image_dict = ammico.find_files(
        path=data_path,
        limit=100,  # Limit the number of files to process (optional)
    )
    return image_dict

def define_model():
    # Configure the externally hosted vision-language model (OpenAI-compatible endpoint).
    # For a self-hosted vLLM server, AMMICO_API_KEY is the value passed to `vllm serve --api-key`.
    os.environ["AMMICO_API_BASE_URL"] = (
        "http://localhost:8000/v1"  # using vllm
        # "http://localhost:11434/v1"  # using ollama
    )
    # os.environ["AMMICO_API_KEY"] = "ollama"  # using ollama
    os.environ["AMMICO_API_KEY"] = "KEY"  # using vllm
    # os.environ["AMMICO_MODEL_ID"] = "qwen2.5vl:3b"  # using ollama
    os.environ["AMMICO_MODEL_ID"] = "Qwen/Qwen2.5-VL-3B-Instruct" # vllm
    # os.environ["AMMICO_PRIVACY_ACK"] = "True"  # using ollama

    model = ammico.InferenceModel()  # reads the AMMICO_API_* environment variables
    return model

def get_instance(model, image_dict):
    # Then, we create an instance of the Python class that handles the image summary and visual question answering tasks:
    image_summary_vqa = ammico.ImageSummaryDetector(summary_model=model, subdict=image_dict)
    return image_summary_vqa

def set_questions():
    # Define the questions for visual question answering
    list_of_questions = [
        "Is there a peacock in the picture, answer with only yes or no?",
        "Is there a peacock in the picture, answer with only a numerical confidence level in percent?",
        # "Is there a bird in the picture, answer with only yes or no?",
        # "Is there a bird in the picture, answer with only a numerical confidence level in percent?",
        # "Is there a peacock in the picture, answer with only yes or no?",
        # "Is there a owl in the picture, answer with only yes or no?",
        # "Is there a falcon in the picture, answer with only yes or no?",
        # "Which animal species is in the picture, answer with only the species name and provide the numerical confidence level in percent?",
    ]  # add or replace with your own questions
    return list_of_questions

def process_images(image_vqa, list_of_questions):
    # Run the analysis on the images using the defined questions
    answers = image_vqa.analyse_images_from_dict(
        analysis_type="questions",
        list_of_questions=list_of_questions,
        is_concise_summary=True,
        is_concise_answer=True,
    )
    return answers

def convert_to_dataframe(filename):
    # To export the results for further processing, convert the image dictionary into a pandas dataframe.
    image_df = ammico.get_dataframe(image_dict)
    image_df.to_csv(f"./{filename}.csv")
    # calculate how many images have a rat in them based on the FIRST COLUMN!!! of the image_dict
    # So the first question is the one that is being evaluated here!
    answers = image_df["vqa"].apply(lambda x: x[0] if x else None)
    answers = [1 if "yes" in str(answer).lower() else 0 for answer in answers]
    rat_count = sum(answers)
    print(f"Number of images with a rat or predator: {rat_count} out of {len(image_df)} images.")

if __name__ == "__main__":
    # get a list of subdirectories in the data path
    # data_path = "../data/biotrove_central_europe_filtered"
    data_path_in = "/home/iulusoy/projects/smart-rodent/smartrodent_experimentsdatasets/biotrove-central-europe-large/biotrove-central-europe-large/imgs"
    subdirs = [d for d in os.listdir(data_path_in) if os.path.isdir(os.path.join(data_path_in, d))]
    print(f"Found {len(subdirs)} subdirectories (species): {subdirs}")
    # for each subdirectory, run the analysis
    for subdir in subdirs:
        print(f"Processing subdirectory: {subdir}")
        data_path = os.path.join(data_path_in, subdir)
        image_dict = load_images(data_path)
        model = define_model()
        image_vqa = get_instance(model, image_dict)
        list_of_questions = set_questions()
        answers = process_images(image_vqa, list_of_questions)
        convert_to_dataframe(subdir)
