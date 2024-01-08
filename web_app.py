import pandas as pd
import gradio as gr

from annoy_db import recommend_top_five_films_bagofwords,recommend_top_five_films_distilbert

def main(user_text, method):
    movie_dataset=pd.read_csv('movies_metadata.csv',dtype=str)
    df=movie_dataset[['id','title','release_date','overview']]
    if method == "Bag of Words":
        return recommend_top_five_films_bagofwords(df,user_text)
    elif method == "DistilBERT":
        return recommend_top_five_films_distilbert(df,user_text)
    else:
        return "Please select a method"

iface = gr.Interface(
    fn=main,
    inputs=[gr.inputs.Textbox(lines=2, placeholder="Enter movie description here..."), 
            gr.inputs.Radio(["Bag of Words", "DistilBERT"], label="Select Method")],
    outputs=[gr.outputs.Dataframe(type="pandas",label="Top 5 Movies")],
    description="Enter text to get top five movie recommendations using Bag of Words or DistilBERT methods."
)

iface.launch(debug=True, share=True)