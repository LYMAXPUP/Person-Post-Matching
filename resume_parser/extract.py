import re
from typing_extensions import final
import hanlp

from .helpers import Exp_Decode, Exp_Extract, extract_skill
from .utils import Extractor


extractor = Extractor()


def get_name(text_list, ner=None):
    entity = ner(text_list)
    for i in entity:
        if i[1] == "PERSON":
            return i[0]
    return ""

def get_current_place(text_list, ner=None):
    for i in range(len(text_list)):
        text_list[i] = re.sub("现居|居住于|现居地", "", text_list[i])
        text_list[i] = re.sub("[：，？！。\|]", "", text_list[i])
    entity = ner(text_list)
    for i in entity:
        if i[1] == "LOCATION":
            return i[0]
    return ""

# 写籍贯的人较少……
def get_native_place(text_list, ner=None):
    for text in text_list:
        if re.search("出生地|籍贯|祖籍", text):
            native_place = re.sub("出生地|籍贯|祖籍", "", text)
            native_place = re.sub("[：，？！。\|]", "", native_place)
            return native_place
    return ""

def get_birthday(text_list):
    for text in text_list:
        if re.search(r'\b(?:\d{4}/\d{2}/\d{2}|\d{4}/\d{2}/\d{2}|\d{4}-\d{2}-\d{2}|\d{4}.\d{2}.\d{2}|\b(?!86)\d{1,2})\b|生日|出生日期|个人信息|年龄|岁', text):
            birthday = re.sub("生日|出生日期|个人信息|年龄|岁", "", text)
            birthday = re.sub("[：，。]", "", birthday)
            return birthday
    return ""

def get_experience(text_list, ner=None):
    for text in text_list:
        if re.search("年|个月", text):
            experience = re.sub("", "", text)
            experience = re.sub("[：，？！。\|]", "", experience)
            return experience
    return ""

def get_gender(text_list):
    for text in text_list:
        if "男" in text:
            return "男"
        elif "女" in text:
            return "女"
        else: continue
    return ""

def get_email(text_list):
    for text in text_list:
        email = extractor.extract_email(text)
        if len(email) > 0:
            return email[0]
    return ""


def get_tel(text_list):
    rex = r'^([0-9]{3,4}-)?[0-9]{7,8}$'
    for text in text_list:
        if re.search(rex, text):
            zipcode = re.search(rex, text)
            return zipcode[0]
    return ""


def get_phone_number(text_list):
    for text in text_list:
        zipcode = extractor.extract_phone_number(text)
        if len(zipcode) > 0:
            return zipcode[0]
    return ""


def get_qq(text_list):
    for text in text_list:
        zipcode = extractor.extract_qq(text)
        if len(zipcode) > 0 and re.search("QQ|qq", text):
            return zipcode[0]
    return ""

def get_id_card(text_list):
    for text in text_list:
        zipcode = extractor.extract_id_card(text)
        if len(zipcode) > 0:
            return zipcode[0]
    return ""



def parse_contact_segment(text_list, segment_dict, model=None):  # list -> dict
    contact_text = segment_dict["contact_segment"]

    #文本预处理
    text = []
    for i in contact_text:
        # 根据换行符分栏
        line_elem = re.split("\n", i)
        for elem in line_elem:
            # 文本清洗
            if elem != "" and elem != " ":
                text.append(extractor.clean_text(elem,
                                remove_parentheses=False, 
                                remove_email=False, 
                                remove_phone_number=False))

    info = []
    for i in text:
        info_elem = re.split("\||\s", i)
        for elem in info_elem:
            if re.sub("[\s]", "", elem) != "":
                info.append(elem)
    contact_text = info
    
    contact_info = {}
    contact_info["name"] = get_name(contact_text, model)
    contact_info["gender"] = get_gender(contact_text)
    contact_info["id_card"] = get_id_card(contact_text)
    contact_info["birthday"] = get_birthday(contact_text)
    contact_info["experience"] = get_experience(contact_text)
    contact_info["current_place"] = get_current_place(contact_text, model)
    contact_info["native_place"] = get_native_place(contact_text)
    contact_info["phone_number"] = get_phone_number(contact_text)
    contact_info["QQ"] = get_qq(contact_text)
    contact_info["email"] = get_email(contact_text)
    contact_info["home_phone_number"] = get_tel(contact_text)
    return contact_info

# -------------------------------- 教育经历 ------------------------------------


def clean(lines):
    text = []
    j = ""
    if len(lines) >= 1:
        del lines[0]
    else: return text
    for i in lines:
        # 去除空行
        if re.sub("\s", "", i) == "":
            continue
        # 去重复行
        elif re.sub("\s", "", i) == j:
            continue
        else:
            j = re.sub("\s", "", i)
            line_elem = re.split("\s[\s]+|\n", i)
            for elem in line_elem:
                if elem != "" and elem != " ":
                    text.append(elem)
    text = sorted(set(text), key=text.index)  # 把重复的元素删除，且保留原有的顺序
    return text


def get_school(text_list):  # str -> str
    rex = r'.*(大学|学院|高中|⼤学|University)'
    for text in text_list:
        if re.search(rex, text):
            school = re.search(rex, text)
            return school.group(0)
    return ""


def get_degree(text_list):
    rex = r'博士|MBA|EMBA|硕士|本科|大专|高中|中专|初中|硕⼠'
    for text in text_list:
        if re.search(rex, text):
            degree = re.search(rex, text)
            return degree.group(0)
    return ""


# 输入为经1，2步处理后的文本数据（默认：lines），输出分块的文本字段。
def parse_education_segment(text_list, segment_dict, edu_extract) -> list:
    education_text = segment_dict['education_segment']
    if len(education_text) <= 1:
        return []

    edu_decode = Exp_Decode()
    # 提取时间，描述，状态
    edu_exp = edu_decode.descrip_extract(education_text)
    # 提取公司，部门，职位，地点，薪水，行业
    edu_info = edu_extract.correct_eduinfo(edu_exp)
    # 综合全部信息
    res = []
    for i in range(len(edu_exp)):
        d = {**edu_exp[i], **edu_info[i]}
        # ?不需要未处理的info项
        # d.pop("info")
        # 提取课程
        courses = edu_extract.courses_extract(d["info"]) + edu_extract.courses_extract(d["description"])
        d["courses"] = courses
        res.append(d.copy())
    return res


# def parse_basic_segment(text_list, segment_dict):  # 最后从提取好的模块中抽取出来
#     text_list

# -------------------------------- 工作经历 ------------------------------------
def parse_work_segment(text_list, segment_dict, work_extract):
    work_text = segment_dict['work_segment']

    if len(work_text) <= 1:
        return []
    work_decode = Exp_Decode()
    # 提取时间，描述，状态
    work_exp = work_decode.descrip_extract(work_text)
    # 提取公司，部门，职位，地点，薪水，行业
    work_info = work_extract.correct_workinfo(work_exp)
    # 综合全部信息
    res = []
    for i in range(len(work_exp)):
        d = {**work_exp[i], **work_info[i]}
        # ?不需要未处理的info项
        # d.pop("info")
        # 提取技能
        skills = []
        for k in d["description"]:
            skills = skills + extract_skill(k)
        d["skills"] = skills
        res.append(d.copy())
    return res


# -------------------------------- 项目和校园经历 ------------------------------------
def parse_project_segment(text_list, segment_dict, project_extract, type = "project_segment"):
    project_text = segment_dict[type]
    project_decode = Exp_Decode()

    if len(project_text) <= 1:
        return []
    project_exp = project_decode.descrip_extract(project_text)
    # 提取公司，部门，职位，地点，薪水，行业
    project_info = project_extract.correct_projectinfo(project_exp)
    # 综合全部信息
    res = []
    for i in range(len(project_exp)):
        d = {**project_exp[i], **project_info[i]}
        # ?不需要未处理的info项
        # d.pop("info")
        # 提取技能
        skills = []
        for k in d["description"]:
            skills = skills + extract_skill(k)
        d["skills"] = skills
        res.append(d.copy())
    return res

# 待完成
def parse_other_segment(text_list, segment_dict):
    other_text = segment_dict['other_segment']
    return other_text


final = {}


def main(text_list, segment_dict, model=None):
    
    # basic_token = parse_basic_segment(text_list, segment_dict)
    if model is None:
        model = model = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
    extractor = Exp_Extract(model)
    final["contact_token"] = parse_contact_segment(text_list, segment_dict, model)
    final["education_token"] = parse_education_segment(text_list, segment_dict, extractor)
    final["work_token"] = parse_work_segment(text_list, segment_dict, extractor)
    final["project_token"] = parse_project_segment(text_list, segment_dict, extractor)
    final["campus_token"] = parse_project_segment(text_list, segment_dict, extractor, "campus_segment")
    final["other_token"] = parse_other_segment(text_list, segment_dict)
    return final
