import pandas as pd
import re


def _df_to_markdown(df):
    return df.to_markdown()


def generate_md_page(tpl_file, scope, out_file):
    with open(tpl_file, 'r') as fh:
        template = fh.read()

    for k, v in scope.items():
        varname = str(k).upper()
        if isinstance(v, (pd.DataFrame, pd.Series)):
            value = _df_to_markdown(v)
        else:
            value = str(v)

        template = template.replace(f'%{varname}%', value)

    unused_tags = re.findall('%[A-Z0-9_-]*%', template)

    for tag in unused_tags:
        print(f'WARNING: Tag {tag} unused')
        template = template.replace(tag, '')

    with open(out_file, 'w') as fh:
        fh.write(template)

    return True
