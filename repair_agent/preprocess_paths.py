import os


def preprocess_paths(agent, project_name, bug_index, filepath):
    workspace = agent.config.workspace_path
    project_dir = os.path.join(workspace, project_name.lower()+"_"+str(bug_index)+"_buggy")
    
    if filepath.endswith(".java"):
        filepath = filepath[:-5]
        filepath = filepath.replace(".", "/")
        filepath += ".java"
    else:
        filepath = filepath.replace(".", "/")
    
    if not os.path.exists(os.path.join(project_dir,filepath)):
        if not os.path.exists(os.path.join(project_dir, "files_index.txt")):
            with open(os.path.join(project_dir, "files_index.txt"), "w") as fit:
                fit.write("\n".join(list_java_files(project_dir)))
            
        with open(os.path.join(project_dir, "files_index.txt")) as fit:
            files_index = [f for f in fit.read().splitlines() if filepath in f]
        
        if len(files_index) == 1:
            filepath = files_index[0]
        elif len(files_index) >= 1:
            raise ValueError("Multiple Candidate Paths. We do not handle this yet!")
        else:
            return "The filepath {} does not exist.".format(filepath)
    return filepath

# NOTE: the live command code uses the preprocess_paths defined in
# autogpt/commands/defects4j.py. This module is a standalone copy kept for
# manual testing only; guard the demo so importing it has no side effects
# (it previously ran on import and referenced an undefined list_java_files).
if __name__ == "__main__":
    class Config():
        def __init__(self, workspace_path):
            self.workspace_path = workspace_path

    class Agent():
        def __init__(self,):
            self.config = Config("auto_gpt_workspace/")

    agent = Agent()

    print(preprocess_paths(agent, "Closure", "10", "src.com.google/javascript/jscomp/CommandLineRunner.java"))