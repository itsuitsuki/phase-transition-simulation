import nbformat

def merge_notebooks(notebook1_path, notebook2_path, output_path):
    with open(notebook1_path, 'r', encoding='utf-8') as nb1_file:
        nb1_content = nbformat.read(nb1_file, as_version=4)

    with open(notebook2_path, 'r', encoding='utf-8') as nb2_file:
        nb2_content = nbformat.read(nb2_file, as_version=4)

    nb1_content['cells'].extend(nb2_content['cells'])

    with open(output_path, 'w', encoding='utf-8') as output_file:
        nbformat.write(nb1_content, output_file)

if __name__ == "__main__":
    notebook1_path = "D:\Desktop\\new_Project\SI140_PJ_Phase_Transition\\the_submission2.ipynb"
    notebook2_path = "D:\Desktop\\new_Project\SI140_PJ_Phase_Transition\part_2_ising.ipynb"
    output_path = "D:\Desktop\\new_Project\SI140_PJ_Phase_Transition\\the_submission3.ipynb"

    merge_notebooks(notebook1_path, notebook2_path, output_path)
