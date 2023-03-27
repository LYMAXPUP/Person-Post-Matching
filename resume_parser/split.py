import re
from collections import defaultdict
import chardet
import yaml

from .utils import Extractor

extractor = Extractor()


def restr(text):
    if isinstance(text,bytes):
        encoding = chardet.detect(text)['encoding']
        text = text.decode(encoding)
    return text


def get_keyword(field):  # 这个函数的就是用技能表中的技能和解析出来的所有文字来匹配，匹配到了就存入csv中
    def get_conf(conf_name, conf_path='resume_parser/confs/config.yaml'):
        CONFS = yaml.load(open(conf_path, 'rb'), Loader=yaml.FullLoader)
        return CONFS[conf_name]

    keyword_dict = {}
    for key, items_of_interest in get_conf(field).items():  # key 就是几个类别 experience:、 platforms:、 database:等
        keyword_dict[key] = items_of_interest
    return keyword_dict


def search_keyword(text, keyword_list):         # 这个函数就是从所有文字中，找切分的关键词
    text = restr(text)
    text_seg = re.split("\n", text)
    for word in keyword_list:
        word = restr(word)
        # if fuzz.ratio(text, word) > 80:
        # if word.title() in text or word.upper() in text or word.capitalize() in text:
        for text in text_seg:
            text = extractor.clean_text(text)
            text = re.sub("[\s\/，、。\&\*]", "", text)
            if (word in text) and (len(text) <= len(word) + 3):
                return True
    return False

# 查看是否有一些基本信息，是的话就跳过
# 为啥跳了……
def validate_text(text):
    # check email in name object
    if text.find("@") > 0 or text.find("github.com") >= 0 or text.find("linkedin.com") >= 0:
        return False
    # check phone number in name object
    cell = re.search("\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}",
                        text)
    if cell:
        return False
    # check hyperlinks in name object
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    if len(urls) > 0:
        return False

    return True

class SegmentsMethod(object):
    def __init__(self):
        self.keyword_dict = get_keyword("segment_keywords")
        self.education_keywords = self.keyword_dict['education_keywords']
        self.work_experience_keywords = self.keyword_dict['work_experience_keywords']
        self.project_keywords = self.keyword_dict['project_keywords']
        self.campus_keywords = self.keyword_dict['campus_keywords']
        self.other_keywords = self.keyword_dict['other_keywords']

        self.user_segment = []  
        self.education_segment = []
        self.work_segment = []
        self.project_segment = []
        self.campus_segment = []
        self.other_segment = []
    
    def check_other_key(self, text:str, type:str)->bool:
        key_type = ['education_keywords', 
            'work_experience_keywords',
            'project_keywords',
            'campus_keywords',
            'other_keywords']
        assert type in key_type
        for i in key_type:
            if i == type:
                continue
            else:
                if search_keyword(text, self.keyword_dict[i]):
                    return True
        return False
    
    def load_segment(self, text_list:list, type:str):
        # Extract Segment
        res = []
        i = 0
        while(i < len(text_list)):
            text = text_list[i]
            if search_keyword(text, self.keyword_dict[type]): # 匹配到了一个关键词之后一直往下搜索
                res.append(text)
                i += 1
                while(i < len(text_list)):
                    text = text_list[i]   # 判断有没有出现其他的关键字，没有的话，就加入
                    if not self.check_other_key(text, type):
                        # if validate_text(text):
                        res.append(text)
                    else:
                        break
                    i += 1
            i += 1
        # 假设所有内容独立，删除分好块的内容
        for k in res:
            text_list.remove(k)
        return res, text_list

    # 这部分弃用
    def load_work_segment(self,text_list):
        for i, text in enumerate(text_list):
            flag = False
            if search_keyword(text, self.work_experience_keywords):
                self.work_segment.append(text)
                i += 1
                flag = True
                while True and i < len(text_list):
                    text = text_list[i]
                    if (not search_keyword(text, self.education_keywords)) and (
                        not search_keyword(text, self.project_keywords)) and (
                            not search_keyword(text, self.other_keywords)) and (
                                not search_keyword(text, self.campus_keywords)):
                        if validate_text(text):
                            self.work_segment.append(text)
                    else:
                        break
                    i += 1
            if flag:
                break
        return self.work_segment

    def load_education_segment(self, text_list):
        # Extract Education Segment
        for i, text in enumerate(text_list):
            flag = False
            if search_keyword(text, self.education_keywords):
                self.education_segment.append(text)
                i += 1
                flag = True
                while True and i < len(text_list):
                    text = text_list[i]
                    if not search_keyword(text, self.work_experience_keywords) and not search_keyword(
                        text, self.project_keywords) and not search_keyword(text, self.other_keywords):
                        if validate_text(text):
                            self.education_segment.append(text)
                    else:
                        break
                    i += 1
            if flag:
                break
        return self.education_segment


    def load_project_segment(self, text_list):
        # Extract Project Segment
        for i, text in enumerate(text_list):
            flag = False
            if search_keyword(text, self.project_keywords):
                self.project_segment.append(text)
                i += 1             # 计数，标记到第几行
                flag = True
                while True and i < len(text_list):  # 如果有keyword，同时小于整个list
                    text = text_list[i]
                    if not search_keyword(text, self.education_keywords) and not search_keyword(
                        text, self.work_experience_keywords) and not search_keyword(text, self.other_keywords):
                        if validate_text(text):
                            self.project_segment.append(text)
                    else:
                        break
                    i += 1

            if flag:
                break
        return self.project_segment

    def load_other_segment(self, text_list):
        # Extract Other Segment
        for i, text in enumerate(text_list):
            flag = False
            if search_keyword(text, self.other_keywords):
                self.other_segment.append(text)
                i += 1
                flag = True
                while True and i < len(text_list):
                    text = text_list[i]
                    if not search_keyword(text, self.education_keywords) and not search_keyword(
                        text, self.work_experience_keywords) and not search_keyword(text, self.project_keywords):
                        if validate_text(text):
                            self.other_segment.append(text)
                    else:
                        break
                    i += 1
            if flag:
                break
        return self.other_segment


def create_pdf_segments(text_list):
    load = SegmentsMethod()
    # work_segment = load.load_work_segment(text_list)
    # education_segment = load.load_education_segment(text_list)
    # project_segment = load.load_project_segment(text_list)
    # other_segment = load.load_other_segment(text_list)
    segment_dict = defaultdict(list)
    segment_dict["education_segment"], text_list = load.load_segment(text_list, "education_keywords")
    segment_dict["work_segment"], text_list = load.load_segment(text_list, "work_experience_keywords")
    segment_dict["project_segment"], text_list = load.load_segment(text_list, "project_keywords")
    segment_dict["campus_segment"], text_list = load.load_segment(text_list, "campus_keywords")
    segment_dict["other_segment"], text_list = load.load_segment(text_list, "other_keywords")
    segment_dict["contact_segment"] = text_list
    print("split done")
    return segment_dict