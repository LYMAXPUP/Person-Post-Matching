import logging
import numpy as np

from ner.ner_predict import get_ner_predict
from match.test import get_match_score, get_text_vec


class Resume:
    def __init__(self, file_id, file_name, jd):
        self.file_id = file_id
        self.file_name = file_name
        self.jd = jd

        self.is_parsed_success = False
        self.name = ""
        self.age = ""
        self.degree = ""
        self.school = ""
        self.telephone = ""
        self.email = ""
        self.gender = 0  # 1 男 0 女
        self.experience = ""
        self.key_words = []
        self.match_score = 0

        self.description_list = []

    def set_basic_para(self, all_info):
        """能从简历中直接读出来的"""
        try:
            self.name = all_info['contact_token']['name']
            self.age = all_info['contact_token']['birthday']
            self.degree = all_info['education_token'][0]['degree']
            self.school = all_info['education_token'][0]['school']
            self.telephone = all_info['contact_token']['phone_number']
            self.email = all_info['contact_token']['email']
            if all_info['contact_token']['gender'] == "男":
                self.gender = 1
            else:
                self.gender = 0
            self.experience = ""
            for project in all_info['project_token']:
                info = ''.join(project['info'])
                project = ''.join(project['description'])
                self.description_list.append(info + project)
            self.is_parsed_success = True
        except KeyError:
            logging.info("解析失败")

    def get_key_words(self):
        key_words = []
        for description in self.description_list:
            key_words.extend(get_ner_predict(description))
        return key_words

    def get_match_score(self, jd):
        # 一份jd可以供多份简历使用，不必重复算
        jd_vec = get_text_vec(jd)
        max_sim_score = 0
        for index, description in enumerate(self.description_list):
            print(self.file_name)
            print("第{0}个长文本".format(index))
            print(description)
            print('-------------')
            max_sim_score = max(max_sim_score, get_match_score(description, jd_vec))
        return np.float64(max_sim_score)
