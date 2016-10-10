import os
import time
import pandas as pd
from IPython.display import display, FileLink, HTML
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook_name):
    """Runs the notebook `notebook_name` (file name with no extension).

    This function executes notebook with name `notebook_name` (no extension)
    and saves the fully executed notebook in a new file appending "-out"
    to the original file name.

    It also displays links to the original and executed notebooks.
    """
    timestamp_cell = "**Executed:** %s\n\n**Duration:** %d seconds."
    nb_name_full = notebook_name + '.ipynb'
    display(FileLink(nb_name_full,
                     result_html_prefix='<b>Source Notebook</b>: '))

    out_path = 'out_notebooks/'
    out_nb_name = out_path + notebook_name + '-out.ipynb'

    nb = nbformat.read(nb_name_full, as_version=4)
    ep = ExecutePreprocessor(timeout=3600)

    start_time = time.time()
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception:
        msg = 'Error executing the notebook "%s".\n\n' % notebook_name
        msg += 'See notebook "%s" for the traceback.' % out_nb_name
        print(msg)
        raise
    finally:
        # Add timestamping cell
        duration = time.time() - start_time
        timestamp_cell = timestamp_cell % (time.ctime(start_time), duration)
        nb['cells'].insert(0, nbformat.v4.new_markdown_cell(timestamp_cell))

        # Write the executed notebook and display link
        nbformat.write(nb, out_nb_name)
        display(FileLink(out_nb_name,
                         result_html_prefix='<b>Output Notebook</b>: '))


def run_notebook_template(notebook_name, remove_out=True,
                          data_ids=['7d', '12d', '17d', '22d', '27d'],
                          ph_sel=None):
    """Run a template ALEX notebook for all the 5 samples.

    Fit results are saved in the folder 'results'.
    For each sample, the evaluated notebook containing both plots
    and text output is saved in the 'out_notebooks' folder.
    """
    timestamp_cell_pattern = "**Executed:** %s\n\n**Duration:** %d seconds."
    # Compute TXT data results file name (removing a previous copy)
    assert ph_sel in ['all-ph', 'Dex', 'DexDem', 'AexAem', 'AND-gate', None]
    ph_sel_suffix = '' if ph_sel is None else '-%s' % ph_sel
    data_fname = 'results/' + notebook_name + '%s.csv' % ph_sel_suffix
    if remove_out and \
       os.path.exists(data_fname):
            os.remove(data_fname)
    nb_name_full = notebook_name + '.ipynb'
    out_path = 'out_notebooks/'

    display(FileLink(nb_name_full,
                     result_html_prefix='<b>Source Notebook</b>: '))
    display(HTML('<ul>'))
    ep = ExecutePreprocessor(timeout=3600)
    for data_id in data_ids:
        nb = nbformat.read(nb_name_full, as_version=4)

        nb['cells'].insert(1, nbformat.v4.new_code_cell('data_id = "%s"' % data_id))
        nb['cells'].insert(1, nbformat.v4.new_code_cell('ph_sel_name = "%s"' % ph_sel))

        out_nb_name = (out_path + notebook_name + '-out%s-%s.ipynb' %
                       (ph_sel_suffix, data_id))

        start_time = time.time()
        try:
            ep.preprocess(nb, {'metadata': {'path': './'}})
        except:
            msg = 'Error executing the notebook "%s".\n\n' % notebook_name
            msg += 'See notebook "%s" for the traceback.' % out_nb_name
            print(msg)
            raise
        finally:
            # Add timestamping cell
            duration = time.time() - start_time
            timestamp_cell = timestamp_cell_pattern % (time.ctime(start_time), duration)
            nb['cells'].insert(0, nbformat.v4.new_markdown_cell(timestamp_cell))

            # Write the executed notebook and display link
            nbformat.write(nb, out_nb_name)
            display(FileLink(out_nb_name,
                result_html_prefix='<li><b>Output Notebook</b> (%s): ' % data_id),
                result_html_suffix='</li>')
    display(HTML('</ul>'))
    display(pd.read_csv(data_fname).set_index('sample').round(4))
    dl_link = FileLink(data_fname, result_html_prefix='<b>Download Data:</b> ')
    display(dl_link)
