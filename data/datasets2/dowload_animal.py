from roboflow import Roboflow 
from pathlib import Path

def dowload_animal_datasets():
  rf = Roboflow(api_key="S7J3y4YEzeoPMG8l5GJU")
  project = rf.workspace("tbitak-bilgem").project("animalclassification-gktyx")
  dataset = project.version(1).download("folder")
  image_path = Path("./AnimalClassification-1")
  train_path = image_path.joinpath("train")
  test_path = image_path.joinpath("test") 
  return train_path, test_path
