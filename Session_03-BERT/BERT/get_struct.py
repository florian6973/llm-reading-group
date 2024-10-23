import re
from collections import defaultdict

# Regular expressions to match class definitions and class instantiations
class_regex = re.compile(r'^class\s+(\w+)\s*\(.*\):')  # Matches "class ClassName(...):"
init_regex = re.compile(r'^\s*def\s+__init__\s*\(.*\):')  # Matches "def __init__(...)"
instantiation_regex = re.compile(r'\bself\.\w+\s*=\s*(\w+)\(')  # Matches "self.attr = ClassName(..."

def extract_class_hierarchy(filename):
    class_hierarchy = defaultdict(list)  # To store class relationships
    current_class = None
    inside_init = False

    with open(filename, 'r') as file:
        for line in file:
            # Check for class definition
            class_match = class_regex.match(line)
            if class_match:
                current_class = class_match.group(1)
                inside_init = False  # Reset init flag when entering a new class
                continue

            # Check if we're inside __init__ method
            if init_regex.match(line):
                inside_init = True
                continue

            # If inside __init__, look for class instantiations
            if inside_init and current_class:
                instantiation_match = instantiation_regex.search(line)
                if instantiation_match:
                    instantiated_class = instantiation_match.group(1)
                    class_hierarchy[current_class].append(instantiated_class)

    return class_hierarchy

def display_hierarchy(class_hierarchy):
    print("Class Hierarchy:")
    for cls, dependencies in class_hierarchy.items():
        if not cls.startswith("Bert"):
            continue
        print(f"{cls} contains the following models: {', '.join([d for d in dependencies if d.startswith('Bert')]) if dependencies else 'None'}")

if __name__ == "__main__":
    # Replace this with the path to your downloaded modeling_bert.py file
    file_path = "modeling_bert.py"
    
    # Extract the class hierarchy
    class_hierarchy = extract_class_hierarchy(file_path)
    
    # Display the class hierarchy
    display_hierarchy(class_hierarchy)
