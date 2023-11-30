import streamlit as st
from sse_forecasting import *
from sse_model_training import *
from sse_ml_utils import *
    
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


style_block = """<style>
                        .header {
                            font-weight: bold; 
                            font-size: 18px;
                            color: #001d2d;
                        }
                        .highlight {
                            background-color: #abe6f0; /* light blue background */
                           
                        }
                        .color-text {
                            color: #00a3f3; /* custom text color */
                            font-weight: bold; /* bold text */
                        }
                        .description {
                            margin-left: 40px; /* increased indent */
                            font-size: 14px; /* smaller text size */
                            color: #555; /* gray text color */
                        }
                        .criteria-description {
                            margin-left: 60px; /* increased indent */
                            font-size: 12px; /* smaller text size */
                            color: #777; /* gray text color */
                        }
              </style>"""


sample_text = """**Sample Text**<br>
          This is sample text: <br /> Sample 1:     <span class="header"> header </span> <br /> Sample 2: <span class="highlight"> highlight </span> <br /> Sample 3:     <span class="color-text"> color text </span> <br /> Sample 4:     <span class="description"> description </span> <br /> Sample 5:     <span class="criteria-description"> criteria-description </span> <br />
""" 


value_col = 'value'
    

def md_text(text, style_block=style_block):
    return st.markdown(style_block +'\n  '+ text, unsafe_allow_html=True)


def page_select():
        """Display a selectbox for page selection and return the selected page."""
        pages = [ "Forecasting Outcomes", "Model Training", "User Guide & Documentation"]
        selected_page = st.selectbox("Choose a page", pages)
        return selected_page
    
css = '''
.stApp {
    
    background-size: cover;
    background-position: center;
}
.stApp > header {
    background-color: transparent;
}

'''
import plotly.graph_objects as go


def generate_markdown(l3_lagging_d, l4_lagging_d):
    # Define a mapping for the color codes
    color_map = {
        'red': '#FF5B59',
        'orange': '#FFB12A'
    }

    # Main header with reduced line height
    markdown_text = "<h1 style='font-size: 16px; line-height: 1;'>Lagging Metrics</h1>\n\n"
    markdown_text += f"Based on the current state, we have identified 5 key lagging areas of SEE Capabilities:"
    for key, color in l3_lagging_d.items():
        # L3 header with specific color and reduced line height
        markdown_text += f"<h2 style='font-size: 16px; color:{color_map[color]}; line-height: 1;'>{key}</h2>\n"

        if key in l4_lagging_d:
            markdown_text += "<div style='margin-left: 18px; line-height: 1'>\n"  # Increased indentation for L4 metrics
            for subkey, subcolor in l4_lagging_d[key].items():
                # L4 entry with specific colored dot and reduced line height
                markdown_text += f"<span style='color:{color_map[subcolor]}; font-size: 15px; line-height: 1;'>●</span> {subkey}<br>\n"
            markdown_text += "</div>\n"

    return markdown_text


def create_beautiful_html_table(data):
    # Define CSS styles
    style = """
    <style>
    html {
      height: 100%;
      width: 90%;
    }
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      font-weight: 100;
      background: linear-gradient(45deg, #49a09d, #5f2c82);
    }
    .container {
      position: relative;
      margin: 0 auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      box-shadow: 0 0 20px rgba(0,0,0,0.1);
      margin: 10px auto;
      background-color:#55608f;
    }
    th, td {
      padding: 15px;
      background-color: rgba(255,255,255,0.3);
      color: #fff;
      text-align: center;
    }
    th {
      background-color:#55608f;
    }
    th.category {
      width: 25%;  /* Equal width for category columns */
    }
    tbody tr:hover {
      background-color: rgba(255,255,255,0.3);
    }
    tbody td {
      position: relative;
    }
    tbody td:hover::before {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      top: -9999px;
      bottom: -9999px;
      background: linear-gradient(45deg, #49a09d, #5f2c82);
      z-index: -1;
    }
    </style>
    """

    # Create the table
    html = '<div class="container"><table>'
    html += '<thead><tr>'
    html += '<th>Metrics</th>'
    for category in data:
        html += f'<th class="category">{category.capitalize()}</th>'
    html += '</tr></thead><tbody>'

    # Extracting the unique keys (columns) from the data
    columns = set()
    for values in data.values():
        columns.update(values.keys())

    # Add data rows for each metric
    for column in columns:
        html += '<tr>'
        html += f'<td>{column.replace("_", " ").capitalize()}</td>'
        for category in data:
            value = data[category].get(column, 'N/A')  # Default to 'N/A' if key is not found
            try:
                html += f'<td>{round(value,2)}</td>'
            except Exception as e: 
                html += f'<td>{value}</td>'
        html += '</tr>'

    # End the table
    html += '</tbody></table></div>'
    return html

# Sample data for the table
data = {
    'speed': {'Sample_size': 1000, 'RMSE validation': 0.05027211056957156, 'R2 validation': 0.8405954898600272},
    'quality': {'Sample_size': 1000, 'RMSE validation': 0.035413290436659674, 'R2 validation': 0.8155393687298187},
    'efficiency': {'Sample_size': 1000, 'RMSE validation': 0.0025851271431862337, 'R2 validation': 0.08499254134080625},
    'team_health': {'Sample_size': 1000, 'RMSE validation': 0.07017866159108531, 'R2 validation': 0.8125019439316351}
}





if 'model_submitted' not in st.session_state:
    st.session_state['model_submitted'] = False

    
def main():
    st.set_page_config(layout="wide")
    if check_password():
        selected_page = page_select()
        if selected_page == "User Guide & Documentation":
            st.write('tbd')
        if selected_page == "Model Training":
            
            
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center; color: #031119;font-size:30px;'>Model Training</h1>", unsafe_allow_html=True)
            #md_text(sample_text)
            
            
            #with c1:
            md_text('<span class="color-text">Core model stats:</span>')
            c1, c2 = st.columns(2)
            with c1:
                st.text('')
                st.text('')
                #st.text('some trained model stats; R2, RMSE, sample size, etc')
                #st.text(data)
                beautiful_html_table = create_beautiful_html_table(data)
                st.markdown(beautiful_html_table, unsafe_allow_html=True)
                st.text('')
                
            with c2:
                imp_data = pd.read_excel('importances_df.xlsx')
                f_metric = st.selectbox("Select a metric", list(['team_health', 'speed', 'quality', 'efficiency']))
                
                imp_plot = create_imp_plot(f_metric, imp_data)
                st.plotly_chart(imp_plot, theme="streamlit", use_container_width = True)
                st.text('')
                with open('importances_df.xlsx', 'rb') as my_file:
                        st.download_button(label = 'Download Feature Importance Scores', data = my_file, file_name = 'feature_importances.xlsx', mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') 
                 
        # Display the HTML content in Streamlit
            
            
            
            st.markdown("<hr/>", unsafe_allow_html=True)
            
            
            st.markdown('<span style="font-size:20px; color:#001d2d;">**To retrain the model:**</span>', unsafe_allow_html=True)
            
            
            
            c1, c2 = st.columns(2)
            with c1:
                with st.form("Train models"):
                    st.markdown('<span style="font-size:18px; color:#00a3f3;">**Step 1:**</span> <span style="font-size:16px;">Upload Outcomes and Capabilities data for model training </span><br> <span style="font-size: 30px;">📥</span>', unsafe_allow_html=True)
            
                    training_X_file_uploaded = False
                    uploaded_training_X = st.file_uploader("Upload Capabilities Excel file for training")
                    if uploaded_training_X is not None:
                        try:
                            message, training_X = read_capabilities_df(uploaded_training_X)
                            if message == "All good!":
                                training_X_file_uploaded = True
                            else:
                                st.warning(message)
                                training_X_file_uploaded = False
                        except Exception as e: 
                            st.text(e)
                            training_X_file_uploaded = False
                            st.warning("File should be in Excel format!")   
    
                    training_Y_file_uploaded = False
                    uploaded_training_Y = st.file_uploader("Upload Outcomes Excel file for training")
                    if uploaded_training_Y is not None:
                        try:
                            message, training_Y = read_outcomes_df(uploaded_training_Y)
                            if message == "All good!": 
                                training_Y_file_uploaded = True
                            else:
                                st.warning(message)
                                training_Y_file_uploaded = False
                        except Exception as e: 
                            training_Y_file_uploaded = False
                            st.warning("File should be in Excel format!")  
    
                    levels_mapping_d, weights_mapping_d, value_range_d, direction_d = read_data_template('metrics_template.xlsx')  
                    st.markdown('<span style="font-size:18px; color:#00a3f3;">**Step 2:**</span> <span style="font-size:16px;">Update Template file with Level weights and mapping (optional) </span><br> <span style="font-size: 30px;">📥</span>', unsafe_allow_html=True)
                    uploaded_template = st.file_uploader("Upload Template Excel file")
                    if uploaded_template is not None:
                        try:
                            levels_mapping_d, weights_mapping_d, value_range_d, direction_d = read_data_template(uploaded_template)
                            uploaded_template = True
                        except Exception as e: 
                            uploaded_template = False
                            st.warning("File should be in Excel format!")  
    
    
                    ready_to_train = False
                    if (training_Y_file_uploaded and training_X_file_uploaded):
                        check1_m, check1 = outcomes_capabilities_match(training_X, training_Y)
                        st.write(check1_m)
                        if check1:
                            check2_m, check2 = template_capabilities_match(training_X, levels_mapping_d)
                            st.write(check2_m)
                            if check2:
                                df_cap = assign_weights_levels(training_X, levels_mapping_d, weights_mapping_d, value_range_d, direction_d)
                                df_cap_pivot = preprocessing_pipeline(df_cap, 'value', team_col='team_id')
                                ready_to_train = True
                    data_submitted = st.form_submit_button("Re-train the model")
                        
                  
            with c2:
                #st.write( st.session_state['model_submitted'] )
                if data_submitted:
                    if ready_to_train:
                        ### Running training pipeline
                        modelling_stats_d = {}
                        outcome_metrics_list = ['speed', 'quality', 'efficiency', 'team_health']
                        
                        if len(outcome_metrics_list)==1:
                            outcome_metrics_list = [outcome_metrics_list]
                        
                        models_updated = {m:False for m in outcome_metrics_list}
                        for outcome_metric in outcome_metrics_list:
                            with st.spinner(f'Training {outcome_metric} model...'):
    
                                df_outcome_selected = training_Y[['team_id', outcome_metric]]
    
                                model_stats = model_training_workflow(df_cap_pivot, df_outcome_selected, outcome_metric, core=False)
                                modelling_stats_d[outcome_metric] = model_stats
                                
                            st.success(f'{outcome_metric} model trained successfully')
                            models_updated[outcome_metric] = True
                          
                        new_model_stats = create_beautiful_html_table(modelling_stats_d)
                        st.markdown(new_model_stats, unsafe_allow_html=True)
                        st.write('')
                        m = st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                            background-color: #00A9F4;
                            color:#ffffff;
                        }
                        div.stButton > button:hover {
                            background-color: #75F0E7;
                            color:#061F79
                            }
                        </style>""", unsafe_allow_html=True)
                        if st.button('Update Core model'):
                            st.session_state['model_submitted'] = True
                            st.write('Core model has been updated')
                              
                            
                            
                    else:
                        st.write('not ready for re-train')
                        
                st.write('')    
                if st.session_state['model_submitted']:
                    st.write('Core model has been updated')
                    
                    
                #if st.button('Update Core Model'):
                    ### add update function
                 #   st.write('Core model has been updated')
    
                
                
        if selected_page == "Forecasting Outcomes":

            team_data = st.file_uploader("Upload team data for analysis")
            st.text('')
            m = st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #00A9F4;
                color:#ffffff;
            }
            div.stButton > button:hover {
                background-color: #75F0E7;
                color:#061F79
                }
            </style>""", unsafe_allow_html=True)
            if st.button('Try with sample team'):
                team_data = 'sample_team.xlsx'
            if team_data is not None:
                ### Reading data
                capabilities_df, outcomes_df = read_sample_data_targets(team_data)
                sample_df  = capabilities_df.copy()
                outcomes_d = prepare_outcomes_d(outcomes_df)
    
                levels_mapping_d, weights_mapping_d, value_range_d, direction_d = read_data_template('metrics_template.xlsx')
    
                ### Data preprocessing
                sample_df = assign_weights_levels(sample_df, levels_mapping_d, weights_mapping_d, value_range_d, direction_d)
    
    
                ### Lagging areas 
                l3_lagging_d, l4_lagging_d = find_lagging_areas(sample_df, value_col, levels_mapping_d, weights_mapping_d)
                fig = plot_pipeline(sample_df, value_col, levels_mapping_d, weights_mapping_d)
                ### Saving table with scores
                output_table = prepare_output_table(sample_df, value_col, levels_mapping_d, weights_mapping_d)
                format_and_save_excel(output_table, 'current_scores_analysis.xlsx', value_col=value_col)
                #wb = load_workbook()
                
    
                ### Forecasting 
                l3_forecast = preprocessing_pipeline(sample_df, 'forecast_1')
                l3_current  = preprocessing_pipeline(sample_df, 'value')
    
                forecasted_outcomes = {}
                outcome_metrics_list = ['speed', 'quality', 'efficiency', 'team_health']
                for outcome in outcome_metrics_list:
                    model_uplift, current_outcome,  forecasted_outcome = forecast_uplift(outcome, outcomes_d, l3_current, l3_forecast)
                    forecasted_outcomes[outcome] = {"% uplift":model_uplift,
                                                    "current outcome":current_outcome,
                                                    "forecasted outcome":forecasted_outcome
                                                    }
    
                st.markdown('<span style="font-size:20px; color:#001d2d;">**1. Current metrics analysis:**</span>', unsafe_allow_html=True)
                st.text('Placeholder for general information on the analyzed team')
                st.text('Missing values, outliers, etc.')

                
                c1, c2 = st.columns([3,2])
                with c1:
                      
                    
                    st.plotly_chart(fig, theme="streamlit", use_container_width = True)
                with c2:
                    st.markdown(generate_markdown(l3_lagging_d, l4_lagging_d), unsafe_allow_html=True)
                    st.markdown('')
                    with open('current_scores_analysis.xlsx', 'rb') as my_file:
                        st.download_button(label = 'Download Full Analysis', data = my_file, file_name = 'scores_analysis.xlsx', mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')  
                st.markdown("<hr/>", unsafe_allow_html=True)
                
                st.markdown('<span style="font-size:20px; color:#001d2d;">**2. Forecasted Outcomes:**</span>', unsafe_allow_html=True)

                c1, c2 = st.columns([2,3])
                with c1:
                    st.text('some stats on accuracy of model fit, etc.')
                with c2:
                    forecast_plot = plot_forecasted_metrics(forecasted_outcomes)
                    st.plotly_chart(forecast_plot, theme="streamlit", use_container_width = True)
                    st.button('Download detailed forecast')
                
                
            
        
if __name__ == "__main__":
    main()
