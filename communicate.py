import json
import os
import shutil

import requests
from flask import Flask, request

from resume_parser.resume import Resume
from resume_parser.main import get_all_info
from ner.ner_predict import get_ner_predict

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route("/resumeFile", methods=["POST"])
def get_resume_file():
    uploadId = ""
    resume_list = []
    jobRequest = ""
    forms = request.form.to_dict()
    jobRequest = forms['jobRequest']
    uploadId = forms['uploadId']
    print("jobRequest: ", jobRequest)
    print("uploadId: ", uploadId)
    print("========================\n")

    if os.path.exists('data'):
        shutil.rmtree('data')
    os.mkdir('data')

    files = request.files.to_dict()
    for file in files.values():
        file_id = file.name
        file_name = file.filename
        resume_list.append(Resume(file_id, file_name, jobRequest))

        data = file.stream.read()
        with open(f'data/{file_name}', 'wb') as f:
            f.write(data)
        if len(data) != 0:
            print(f"get {file_name} success.")

    send_info(uploadId, resume_list)
    return "get resume success."


def get_resumes_info(resume_list):
    for resume in resume_list:
        all_info = get_all_info(f"data/{resume.file_name}")
        resume.set_basic_para(all_info)

    # 按格式输出
    resumes_info_list = []
    for resume in resume_list:
        resumeInfo = {
            "id": resume.file_id,
            "isParsedSuccess": resume.is_parsed_success,
            "basicInfo": {
                    "name": resume.name,
                    "age": resume.age,
                    "degree": resume.degree,
                    "school": resume.school,
                    "telephone": resume.telephone,
                    "email": resume.email,
                    "gender": resume.gender,
                    "experience": resume.experience,
                },
            "keyWords": resume.get_key_words(),
            "matchScore": resume.get_match_score(resume.jd)
        }
        resumes_info_list.append(resumeInfo)

    return resumes_info_list


def send_info(uploadId, resume_list):
    info = {
        "uploadId": uploadId,
        "resumesInfo": get_resumes_info(resume_list)
    }
    response = requests.post("http://39.101.65.71:8080/resumeInfo",
                          headers={"Content-type": "application/json"},
                          data=json.dumps(info, ensure_ascii=False, indent=4).encode("utf-8"))
    print(info)
    print(response.status_code)
    print(response.text)

@app.route("/job", methods=["POST"])
def get_job_key_words():
    jobRequest = ""
    forms = request.form.to_dict()
    jobRequest = forms['jobRequest']
    print("jobReques=", jobRequest)
    print("===============")
    key_words = get_ner_predict(jobRequest)
    return key_words


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1234)

