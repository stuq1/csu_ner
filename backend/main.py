from fastapi import FastAPI
import uvicorn

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

def parser_predict(pred_res):
    res = []

    for i in range(0, len(pred_res)):
        word = pred_res[i]["word"]
        label_ind = int(pred_res[i]['entity'][6:])
        label = ru_label_names[label_ind]

        if (word[:2] == '##'):
            res[len(res) - 1]["word"] = res[len(res) - 1]["word"] + word[2:]
        else:
            res.append({"word": word, "label": label})

    return res

ru_label_names = ['O', 'I-AGE', 'B-AGE', 'B-AWARD', 'I-AWARD', 'B-CITY', 'I-CITY', 'B-COUNTRY', 'I-COUNTRY', 'B-CRIME', 'I-CRIME', 'B-DATE', 'I-DATE', 'B-DISEASE', 'I-DISEASE', 'B-DISTRICT', 'I-DISTRICT', 'B-EVENT', 'I-EVENT', 'B-FACILITY', 'I-FACILITY', 'B-FAMILY', 'I-FAMILY', 'B-IDEOLOGY', 'I-IDEOLOGY', 'B-LANGUAGE', 'I-LAW', 'B-LAW', 'B-LOCATION', 'I-LOCATION', 'B-MONEY', 'I-MONEY', 'B-NATIONALITY', 'I-NATIONALITY', 'B-NUMBER', 'I-NUMBER', 'B-ORDINAL', 'I-ORDINAL', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-PENALTY', 'I-PENALTY', 'B-PERCENT', 'I-PERCENT', 'B-PERSON', 'I-PERSON', 'I-PRODUCT', 'B-PRODUCT', 'B-PROFESSION', 'I-PROFESSION', 'B-RELIGION', 'I-RELIGION', 'B-STATE_OR_PROVINCE', 'I-STATE_OR_PROVINCE', 'B-TIME', 'I-TIME', 'B-WORK_OF_ART', 'I-WORK_OF_ART']

model = AutoModelForTokenClassification.from_pretrained("./model", num_labels=len(ru_label_names))
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

ner_pipeline = pipeline('ner', model=model, tokenizer=tokenizer, device=0)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/predict")
async def predict(text: str):
    return parser_predict(ner_pipeline(text))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8090)
