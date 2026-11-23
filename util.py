import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path
from html import escape
import re
import unicodedata
from urllib.parse import quote


# --- Metadata extraction ----
def infer_metadata(df, cat_unique_threshold=0.05, max_cat_unique=20):
    
    metadata = {"columns": {}}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        unique_vals = series.dropna().unique()
        n_unique = len(unique_vals)

        # ---- Boolean columns → categorical ----
        if pd.api.types.is_bool_dtype(series):
            sdtype = "categorical"

        # ---- Integer columns ----
        elif pd.api.types.is_integer_dtype(series):
            # encoded boolean values (0/1)
            if set(unique_vals).issubset({0, 1}):
                sdtype = "categorical"
            # encoded categorical values
            elif (n_unique <= max_cat_unique) and (n_unique / n_rows <= cat_unique_threshold):
                sdtype = "categorical"
            else:
                sdtype = "numerical"

        # ---- Float columns ----
        elif pd.api.types.is_float_dtype(series):
            if (n_unique <= max_cat_unique) and (n_unique / n_rows <= cat_unique_threshold):
                sdtype = "categorical"
            else:
                sdtype = "numerical"

        # ---- Datetime ----
        elif pd.api.types.is_datetime64_any_dtype(series):
            sdtype = "datetime"

        # ---- Object/string ----
        else:
            sdtype = "categorical"

        metadata["columns"][col] = {"sdtype": sdtype}

    return metadata

def modify_generate_metadata(dataframe, metadata, verbose=True):
    """
    Set metadata to default sdtype properties based on column type.
    """
    # Add default sdtype property values
    for col, info in metadata["columns"].items():

        sdtype = info["sdtype"]

        # numerical → computer_representation
        if sdtype == "numerical":
            if pd.api.types.is_integer_dtype(dataframe[col]):
                info["computer_representation"] = "Int64"
            else:
                info["computer_representation"] = "Float"
      
        # datetime → date_time_format
        elif sdtype == "datetime":
            info["date_time_format"] = "%Y-%m-%d"

        # id → regex_format (optional, if you decide to mark ID columns)
        elif sdtype == "id":
            info["regex_format"] = "[0-9]{10}"

    if verbose:
        print(json.dumps(metadata, indent=4))

    return metadata

# --- Fidelity Report -----

def norm_label(value):
    """Normalize a category label so 1, 1.0 and '1' are the same category."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f.is_integer() else str(f)


def ordinal_rank_map(values, order_by="alphabetical"):
    """Map ordinal categories to ranks 0..k-1.
    order_by='numerical_value' sorts numbers numerically (2 < 10); otherwise labels are sorted as text.
    """
    labels = sorted({norm_label(v) for v in values})
    if order_by == "numerical_value":
        def key(s):
            try:
                return (0, float(s), s)
            except ValueError:
                return (1, 0.0, s) # non-numeric labels go last
        labels = sorted(labels, key=key)
    return {lab: i for i, lab in enumerate(labels)}


def encode_ordinals(dfs, ord_specs):
    """Replace ordinal columns with integer ranks, using the SAME mapping in every DataFrame.
    dfs: list of DataFrames (e.g. [train, test, synthetic])
    ord_specs : list of (column, domain, order_by)
    Returns copies of the DataFrames. Categories seen in any dataset but missing from the
    domain are inserted at their sorted position, so every value gets a rank.
    """
    out = [df.copy() for df in dfs]
    for col, domain, order_by in ord_specs:
        values = list(domain) + [v for df in out if col in df.columns for v in df[col].unique()]
        mapping = ordinal_rank_map(values, order_by)
        for df in out:
            if col in df.columns:
                df[col] = df[col].map(lambda v: mapping[norm_label(v)]).astype(int)
    return out

def merge_figurewidgets(
    fig_widgets,
    max_cols=3,
    subplot_height=300,
    subplot_width= 350,
    horizontal_spacing=0.1,
    vertical_spacing=0.15,
    shared_legend=True,
):
    """
    Merge multiple go.FigureWidget into a single go.Figure
    Clean grid: no subplot titles, all x-axis labels visible, proper spacing
    """

    n = len(fig_widgets)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    final_fig = make_subplots(
        rows=rows,
        cols=cols,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    for idx, fw in enumerate(fig_widgets):
        row = idx // cols + 1
        col = idx % cols + 1

        # Recreate traces
        for trace in fw.to_plotly_json()['data']:
            showlegend = trace.get('showlegend', True)
            if shared_legend and not (row == 1 and col == 1):
                showlegend = False

            new_trace = go.Figure(trace).data[0]
            new_trace.showlegend = showlegend
            final_fig.add_trace(new_trace, row=row, col=col)

        # Copy x-axis title if exists, else use blank
        x_title = fw.layout.xaxis.title.text if 'xaxis' in fw.layout and fw.layout.xaxis.title.text else ""
        final_fig.update_xaxes(title_text=x_title, automargin=True, showticklabels=True, row=row, col=col)

        # Copy y-axis title if exists
        y_title = fw.layout.yaxis.title.text if 'yaxis' in fw.layout and fw.layout.yaxis.title.text else ""
        final_fig.update_yaxes(title_text=y_title, automargin=True, showticklabels=True, row=row, col=col)
       
    # Set overall figure size
    final_fig.update_layout(
        height=subplot_height * rows,
        width=subplot_width * cols,
        showlegend=shared_legend,
        margin=dict(t=50, l=60, r=40, b=60),
    )

    return final_fig

# ---- Privacy Report -----
def sample_attack_df(train_set, test_set, state=1):
    """Attacker's dataset: same size as the training data, 50% members and 50% non-members,
    in random order. Returns members, non_members, attack_df, labels (1 = member)."""
    
    t_frac = train_set.shape[0] / (train_set.shape[0] + test_set.shape[0])
    members = train_set.sample(frac=t_frac, random_state=state)
    non_members = test_set.sample(frac=t_frac, random_state=state)

    attack_df = pd.concat([members, non_members], ignore_index=True)
    labels = np.concatenate([np.ones(len(members), dtype='int8'), np.zeros(len(non_members), dtype='int8')])

    perm = np.random.default_rng(state).permutation(len(attack_df))
    return members, non_members, attack_df.iloc[perm].reset_index(drop=True), labels[perm]

def record_matching(distance_matrix, closest_idx, min_dcr=0.0, labels=['Df1_index', 'Df2_index', 'Distance']):
    
    n = distance_matrix.shape[0]  
    closest_distance = distance_matrix[np.arange(n), closest_idx]
    match_df = pd.DataFrame({
            labels[0]: np.arange(n),
            labels[1]: closest_idx,    
            labels[2]: closest_distance # Distance to closest record. 
        })        
    pair_min_dcr_dist = match_df[match_df[labels[2]] <= min_dcr]
    
    return match_df, pair_min_dcr_dist


def pvalue_format(pvalue):
    
    if pvalue < 0.001:
        p = "< 0.001"
    elif pvalue < 0.05:
        p = "< 0.05"
    else:
        p = np.round(pvalue, decimals=3)
    return p

# --- Utility Report -----

def safe_filename(name, max_len=100):
    """Turn any text into a safe file name (without extension).
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()  # 'Ödem' -> 'Odem'
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")
    return s[:max_len] or "file"

def get_estimator_options(train_real_df, target_name, cont_var, ord_var, cat_var ):
    """Determine estimator options based on the target variable type."""
    target = train_real_df[target_name]
    
    # Copy the variable lists
    cont_names = [n for n, v in cont_var]
    ord_names = [n for n, v in ord_var]
    cat_names = [n for n, v in cat_var]

    # Handle binary categorical targets
    if target_name in cat_names and train_real_df[target_name].nunique() == 2:
        cat_names.remove(target_name)
        ord_names.append(target_name)

    if target_name in cont_names:
        options = ['Linear Regression', 'KNN Regressor', 'Random Forest Regressor']
    elif target_name in ord_names:
        if target.nunique() == 2:
            options = ['Logistic Regression', 'KNN Classifier', 'Random Forest Classifier']
        else:
            options = ['KNN Classifier', 'Multi Logistic Regression', 'Random Forest Classifier']
    elif target_name in cat_names:
        options = ['KNN Classifier', 'Multi Logistic Regression', 'Random Forest Classifier']
    else:
        options = []

    return options, cont_names, ord_names, cat_names

# Summary Statistics
def summarize_continuous(df, variables, dataset_name):
    rows = []
    for var in variables:
        mean, std = df[var].mean(), df[var].std()
        median, q1, q3 = df[var].median(), df[var].quantile(0.25), df[var].quantile(0.75)
        min_val, max_val = df[var].min(), df[var].max()
        stats = [
            ("Mean ± SD", f"{mean:.1f} ± {std:.1f}"),
            ("Median (IQR)", f"{median:.1f} ({q1:.1f}–{q3:.1f})"),
            ("Min–Max", f"{min_val:.1f}–{max_val:.1f}")
        ]
        for label, value in stats:
            rows.append({
                "Variable": var,
                "Statistic / Category": label,
                dataset_name: value
            })
    return pd.DataFrame(rows)


def summarize_categorical(df, variables, dataset_name):
    rows = []
    for var in variables:
        counts = df[var].value_counts(dropna=False)
        total = counts.sum()
        for cat, count in counts.items():
            pct = count / total * 100
            rows.append({
                "Variable": var,
                "Statistic / Category": str(cat),
                dataset_name: f"{count} ({pct:.1f}%)"
            })
    return pd.DataFrame(rows)

def prepare_table_grouped_background(df):
    rows, colors = [], []
    group_colors = ["#f8f9fa", "#f1f3f5"] 
    color_index = 0

    for var in df["Variable"].unique():
        sub = df[df["Variable"] == var]
        if var.startswith("**"):  # section header
            rows.append([var.replace("**", ""), "", "", ""])
            colors.append("#e5ecf6")  # Plotly header-like color
        else:
            first = True
            for _, r in sub.iterrows():
                rows.append([
                    var if first else "",
                    r["Statistic / Category"],
                    r["Real"],
                    r["Synthetic"]
                ])
                colors.append(group_colors[color_index])
                first = False
            color_index = 1 - color_index
    df_out = pd.DataFrame(rows, columns=["Variable", "Statistic / Category", "Real", "Synthetic"])
    return df_out, colors

def plot_summary_table(df, row_colors):   
    row_height = 28  
    base_height = 90  # for title and margins
    fig_height = base_height + row_height * len(df)
    
    fig = go.Figure(
        data=[go.Table(
            columnwidth=[275, 180, 180, 180],  
            header=dict(
                values= ['<b>'+col+'</b>' for col in df.columns],
                fill_color="#c8d3e5",
                align="center",
                font=dict(size=14),
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color=[row_colors],
                align=["center", "center", "center", "center"],
                height=28,
                font=dict(size=12, color="black"),               
            )
        )]
    )

    fig.update_layout(
        width=950,
        height=fig_height,        
        margin=dict(t=10, b=5, l=5, r=5)
    )

    return fig

def style_dict_to_css(style):
    if not style:
        return ""
    if isinstance(style, dict):
        return "; ".join(f"{k}: {v}" for k, v in style.items())
    return style     

def build_report(input_dir, sections, output_file="report.html", title="Analysis Report", title_style=None):
   
    input_dir = Path(input_dir)
    output_dir = Path(output_file)
    title_css = style_dict_to_css(title_style)
    body_sections = []

    for i, sec in enumerate(sections, start=1):
        section_id = f"section-{i}"
        section_html = ""

        # ---- Top text sections ----
        if sec.get("type") == "text":
            section_html += f"<div style='font-size:20px; font-weight:bold;'>{escape(sec.get('sub_title', ''))}</div>"
            section_html += f"{sec.get('content', '')}"
            body_sections.append(f'<section id="{section_id}">{section_html}</section>')
            continue

        # ---- Main sections ----
        if sec.get("type") == "section":
            
            # Title (only if present)
            if sec.get("title"):
                section_html += f"<h2>{escape(sec['title'])}</h2>"
                
            # Subtitle
            if "sub_title" in sec and sec["sub_title"]:
                section_html += f"<div style='color:#0f736b; font-size:18px; font-weight:bold; margin-top:10px; margin-bottom:10px;'>{escape(sec.get('sub_title', ''))}</div>"

            # Explanatory text
            if "text" in sec and sec["text"]:
                section_html += f"<p>{escape(sec['text'])}</p>"
                
            if "content" in sec and sec["content"]:
                section_html += f"{sec['content']}"

            # HTML tables
            for html_file in sec.get("html_files", []):
                path = input_dir / html_file
                if not path.exists():
                    raise FileNotFoundError(f"Missing HTML file: {path}")
                html_content = path.read_text(encoding="utf-8")
                section_html += f"<div>{html_content}</div>"

            # Images
            images = sec.get("image_files", [])
            if len(images) == 2:
                # Section with 2 images side by side
                section_html += (
                    '<div style="display:flex; flex-wrap:wrap; gap:3px; align-items:flex-start;">'
                )
                for img_file in images:
                    path = input_dir / img_file
                    if not path.exists():
                        raise FileNotFoundError(f"Missing image file: {path}")
                    # Use relative path from HTML file
                    rel_path = path.relative_to(output_dir.parent)
                    section_html += (
                        f'<div style="flex: 0 0 45%; min-width:200px;">'
                        f'<img src="{quote(rel_path.as_posix())}" style="width:100%; max-width:600px; height:auto; display:block;">'
                        '</div>'
                    )
                section_html += '</div>'                
            else:
                # Regular images stacked
                for img_file in images:
                    path = input_dir / img_file
                    if not path.exists():
                        raise FileNotFoundError(f"Missing image file: {path}")
                        
                    rel_path = path.relative_to(output_dir.parent)                    
                    section_html += f'<div style="margin-top:20px;"><img src="{quote(rel_path.as_posix())}" style="max-width:100%; display:block;"></div>'

            # Wrap section with page-break
            page_break = sec.get("page_break", False)
            style = "page-break-after: always;" if page_break else ""
            body_sections.append(f'<section id="{section_id}" style="{style}">{section_html}</section>')

        else:
            raise ValueError(f"Unknown section type: {sec.get('type')}")

    # ---- Final HTML ----
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{escape(title)}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 15px; line-height: 1.5; text-align: left; }}
            h1 {{ text-align: left; margin-bottom: 15px; }}
            h2 {{ color: darkblue; font-weight:bold; margin-bottom: 10px;}}
            section {{ margin-bottom: 10px; }}
            img {{ vertical-align: top; }}
            table {{ border-collapse: collapse; width: 100%; margin-top:10px; }}
            table, th, td {{ border: 1px solid black; }}
            th, td {{ padding: 5px; text-align: left; }}
        </style>
    </head>
    <body>
        <h1 style="{title_css}">{escape(title)}</h1>
        {"".join(body_sections)}
    </body>
    </html>
    """

    Path(output_file).write_text(report_html, encoding="utf-8")
    return output_file

def clean_directory(keep_files, input_dir):
    # Remove from the Utility folder all files that are not included in the report
    directory = Path(input_dir)      
        
    for item in directory.iterdir():
        if item.is_file() and item.name not in keep_files:
                item.unlink()  





    