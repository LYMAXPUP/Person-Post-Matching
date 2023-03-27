import re
import time
import chardet
import numpy as np
from datetime import datetime
from pyhanlp import *

from .utils import Extractor

extractor = Extractor()

# ------------------------------------- 日期 ----------------------------------------------
# 匹配正则表达式
matches = {
    # xxxx年xx(月)-xx月
    1: (r'\d{4}%s\d{1,2}%s%s\d{1,2}%s', '%%y%s%%m%s%s%%m%s'),
    2: (r'\d{4}%s\d{1,2}%s%s', '%%y%s%%m%s%s'),
    3: (r'\d{2}%s\d{1,2}%s%s', '%%y%s%%m%s%s'),

    # xxxx(年)-xxxx年
    4: (r'\d{4}%s%s\d{4}%s', '%%y%s%s%%y%s'),
    5: (r'\d{4}%s%s', '%%y%s%s'),

    # xxxx年xx月xx日至今
    6: (r'\d{4}%s\d{1,2}%s\d{1,2}%s', '%%y%s%%m%s%%d%s'),

    # xxxx年xx月xx日
    7: (r'\d{4}%s\d{1,2}%s\d{1,2}%s', '%%Y%s%%m%s%%d%s'),

    # xxxx年xx月
    8: (r'\d{4}%s\d{1,2}%s', '%%Y%s%%m%s')
}

# 正则中的%s分割
splits = [
    {1: [('年', '月', '[-\~—至]+', '月'), ('年', '', '[-\~—至]+', '月')]},
    {2: [('年', '月', '[ -\~\—]*至今'), ('-', '', '[ -\~\—]*至今'),
         ('\.', '', '[ -\~\—]*至今'), ('\/', '', '[ -\~\—]*至今')]},
    {3: [('年', '月', '[ -\~\—]*至今')]},

    {4: [('年', '[-\~—至]+', '年'), ('', '[-\~—至]+', '年')]},
    {5: [('年', '[ -\~\—]*至今')]},
    {6: [('年', '月', '日[ -\~\—]*至今'), ('-', '-', '[ -\~\—]*至今'),
         ('\.', '\.', '[ -\~\—]*至今'), ('\/', '\/', '[ -\~\—]*至今')]},

    {7: [('年', '月', '日'), ('\/', '\/', ''), ('\.', '\.', ''), ('-', '-', '')]},

    {8: [('年', '月'), ('\/', ''), ('\.', ''), ('-', '')]}
]
delimiter = ", "


class TimeFinder(object):

    def __init__(self):
        self.match_item = []
        self.init_match_item()

    def init_match_item(self):
        # 构建穷举正则匹配公式，及提取的字符串转datetime格式映射
        for item in splits:
            for num, value in item.items():
                match = matches[num]
                for sp in value:
                    tmp = []
                    for m in match:
                        tmp.append(m % sp)
                    self.match_item.append(tuple(tmp))

    def parse_time_text(self, text, still_active=False):
        # 处理非xxxx年xx月类型
        res = []
        year0 = re.search('\d{4}', text).group()
        text = re.sub(year0, '', text, count=1)
        pre_year = time.strftime("%Y", time.localtime())
        pre_mon = time.strftime("%m", time.localtime())
        if len(year0) == 2: 
            year0 = '20' + year0
        # 处理至今类型
        if '今' in text:
            if re.search('\d{1,2}', text) is not None:
                months = re.findall('\d{1,2}', text)
                res.append({'year': year0, 'month': months[0]})
            else:
                res.append({'year': year0, 'month': ''})
            res.append({'year': pre_year, 'month': pre_mon})
            still_active = True
        # 处理
        elif '月' in text or '/' in text or '.' in text:
            months = re.findall('\d{1,2}', text)
            res.append({'year': year0, 'month': months[0]})
            if int(months[1]) > int(pre_mon):
                still_active = True
            res.append({'year': year0, 'month': months[1]})
        # 处理xxxx-xxxx年类型
        else:
            res.append({'year': year0, 'month': ''})
            year1 = re.search('\d{2,4}', text).group()
            if len(year1) == 2: 
                year1 = '20' + year1
            if int(year1) > int(pre_year):
                still_active = True
            res.append({'year': year1, 'month': ''})
        return res, still_active

    def find_time(self, text):
        # 格式化text为str类型
        if isinstance(text, bytes):
            encoding = chardet.detect(text)['encoding']
            text = text.decode(encoding)
        
        # 文本初步清洗
        text = re.sub("[\s\$\^\#\&\*()\%\+\=]", "", text)

        res = []
        pattern = '|'.join([x[0] for x in self.match_item])
        pattern = pattern
        match_list = re.findall(pattern, text)

        still_active = False
        pre_year = time.strftime("%Y", time.localtime())
        pre_mon = time.strftime("%m", time.localtime())

        if not match_list:
            return None
        for match in match_list:
            for item in self.match_item:
                flag = 0
                if (len(item) == 2):
                    try:
                        # 年-月转换 （待改进）
                        date = datetime.strptime(match,item[1].replace('\\',''))
                        if date.year >= 1970 and date.year <= 2030:
                            res.append({'year': str(date.year),
                                        'month': str(date.month)})
                            if date.year > int(pre_year) or (
                                    date.year == int(pre_year) and
                                    date.month > int(pre_mon)):
                                still_active = True
                            flag = 1
                            break
                    except Exception as e:
                        continue
            # 如果出现了时间段的描述，则优先选取时间段，并跳出循环
            if flag == 0: 
              res, still_active = self.parse_time_text(match)
            # 至多寻找两个日期
            if len(res) >= 2:
                break
        if not res or len(res) <= 1:
            return None
        if float(res[0]["year"]) > float(res[1]["year"]):
            return None
        elif float(res[0]["year"]) < 1980 or float(res[1]["year"]) > 2030:
            return None
        if res[0]["month"] != "":
            if float(res[0]["month"]) > 12:
                return None
        if res[1]["month"] != "":
            if float(res[1]["month"]) > 12:
                return None
        return res, still_active

# --------------------------------------- 经历分块 ----------------------------------------------
class Exp_Decode(object):
    def __init__(self):
        self.timefinder = TimeFinder()
        self.para_sign = "[。，；、\,\！\？\?\!]|\d.[\s]*[\u4e00-\u9fa5]"

    # 分段得分函数，预测类型（信息vs.描述）
    def score(self, text:str, ltype:str="none", date_num:int=0)->str:
        info = 0
        des = 0
        # 日期分
        if date_num: info = info + 8 * date_num
        # 文本分
        info = info + len(re.findall("[\t ]", text)) * 2
        if (len(text) <= 10 and 
                re.search("[。；！？，、]|\d.[\s]*[\u4e00-\u9fa5]", text) is None): 
            info = info + 8
        elif len(text) >= 40: des = des + 8
        # 标点分
        des = des + len(re.findall("[。；！？]", text)) * 10
        des = des + len(re.findall("[，、]", text)) * 3
        # 格式分
        if ltype == "info":
            info = info + 5
        elif ltype == "des":
            des = des + 5
        # 返回类型
        if des > info: 
            return "des"
        elif info > des: 
            return "info"
        else: 
            return ltype

    # 给定起点终点（不含终点），处理/纠错某一段经历
    def unit_extract(self, text:list, startl:int, endl:int, bfd:int):
        assert startl <= bfd
        assert bfd <= endl
        # 写入经历
        exp = {}
        # 分为info（信息实体）和decription
        exp["info"] = []
        exp["description"] = []
        exp["start_year"] = ""
        exp["start_month"] = ""
        exp["end_year"] = ""
        exp["end_month"] = ""
        exp["still_active"] = False

        current_line = endl

        # 提取info
        ltype = "info"
        for j in range(bfd, startl-1, -1):
            if j == bfd:
                period, exp["still_active"] = self.timefinder.find_time(text[j])
                if len(period) == 2:
                    exp["start_year"] = period[0]["year"]
                    exp["start_month"] = period[0]["month"]
                    exp["end_year"] = period[1]["year"]
                    exp["end_month"] = period[1]["month"]
                else:
                    exp["start_year"] = period[0]["year"]
                    exp["start_month"] = period[0]["month"]
                    exp["end_year"] = period[0]["year"]
                    exp["end_month"] = period[0]["month"]
                exp["info"].append(text[j]) # 以防日期外还有别的信息
            elif (self.score(text[j], ltype) == "info" or 
                    self.score(text[j], ltype) == "none"):
                exp["info"].insert(0, text[j])
            else: break

        # 提取description
        ltype = "none"
        flag = True
        for j in range(bfd+1, endl):
            if j >= endl - (bfd - startl):
                ltype = "info"
                if flag: current_line = j 

            if self.score(text[j], ltype) == "info" and ltype == "none":
                exp["info"].append(text[j])
            elif (self.score(text[j], ltype) == "des" or
                    self.score(text[j], ltype) == "none"):
                ltype = "des"
                exp["description"].append(text[j])
                flag = True
            else:
                if flag:
                    current_line = j
                    flag = False
                continue

        return exp, current_line        

    # 处理无日期的经历
    def seq_extract(self, text:list):
        res = []
        # 写入经历
        exp = {}
        # 分为info（信息实体）和decription
        exp["info"] = []
        exp["description"] = []
        exp["start_year"] = ""
        exp["start_month"] = ""
        exp["end_year"] = ""
        exp["end_month"] = ""
        exp["still_active"] = False

        currentl = 0
        endl = len(text)

        # 提取经历
        ltype = "info"
        for j in range(0, endl):
            if ltype == "info":
                if (self.score(text[j], ltype, 0) == "info" or self.score(text[j], ltype, 0) == "none"):
                    exp["info"].append(text[j])
                else:
                    exp["description"].append(text[j])
                    ltype = "des"
            else:
                if (self.score(text[j], ltype, 0) == "des" or self.score(text[j], ltype, 0) == "none"):
                    exp["description"].append(text[j])
                else:
                    # 重启exp
                    res.append(exp.copy())
                    exp["info"] = []
                    exp["description"] = []
                    ltype = "info"
        res.append(exp.copy())
        return res

    # 输入为经1，2步处理后的文本数据（默认：lines），输出分块的文本字段。
    def descrip_extract(self, lines:list)->list:
        text = []
        # 预处理pdf转txt的数据
        j = ""
        for i in lines:
            # 去除空行
            if re.sub("\s", "", i) == "":
                continue
            # 去重复行
            elif re.sub("\s", "", i) == j:
                continue
            else:
            # 根据简历书写特点，分开每一行的组件
                j = re.sub("\s", "", i)
            # 根据换行符分栏
                line_elem = re.split("\n", i)
                for elem in line_elem:
                    # 文本清洗
                    if elem != "" and elem != " ":
                        text.append(extractor.clean_text(elem,remove_parentheses=False))
        # 去除第一个元素(即工作经历、社会经历的标题)
        if len(text) >= 1:
            del text[0]
        # 计算时间节点
        date_pts = []
        for i in range(len(text)):
            if self.timefinder.find_time(text[i]) is not None:
                date_num = len(self.timefinder.find_time(text[i])[0])
                if self.score(text[i], "none", date_num) == "info":
                    date_pts.append(i)
            else: continue
        date_pts.append(len(text))
    
        res = []
        # 提取经历
        if len(date_pts) > 1:
            startl = 0
            for i in range(len(date_pts)-1):
                endl = date_pts[i+1]
                exp, startl = self.unit_extract(text, startl, endl, date_pts[i])
                res.append(exp.copy())
        else:
            res = self.seq_extract(text)
        return res

# --------------------------------------- 信息提取 ----------------------------------------------
class Exp_Extract(object):
    def __init__(self, model):
        """model: used named entity recognition model"""
        self.ner = model

    def info_process(self, exp:dict)->list:
        """preprocess the info of a dict of experience"""
        info = []
        for i in exp["info"]:
            info_elem = re.split("\||\s", i)
            for elem in info_elem:
                if re.sub("[\s]", "", elem) != "":
                    info.append(elem)
        exp["info"] = info
        return info

    # 提取教育机构
    def get_school(self, text:str)->str:  # str -> str
        rex = r'.*(大学|学院|高中|⼤学|University)'
        if re.search(rex, text):
            school = re.search(rex, text)
            return school.group(0)
        return ""
    
    # 提取学历
    def get_degree(self, text:str)->str:
        rex = r'博士|MBA|EMBA|硕士|本科|大专|高中|中专|初中|硕⼠'
        if re.search(rex, text):
            degree = re.search(rex, text)
            return degree.group(0)
        return ""

    # 提取系、学院
    def get_department(self, text:str)->str:
        rex = r"学院|系|研究所"
        if re.search(rex, text):
            degree = re.search(rex, text)
            return degree.group(0)
        return ""

    # 提取专业
    def get_major(self, text:str)->str:
        rex = "专业|主修专业"
        if re.search(rex, text):
            degree = re.search(rex, text)
            return degree.group(0)
        return ""
    
    # 提GPA
    def get_GPA(self, text:str)->str:
        key = "绩点|学分绩|GPA|加权平均分"
        # 等级制、百分制
        rex = "[\d]{1}\.[\d]{1,3}|\d{2}"
        if re.search(key, text):
            if re.search(rex, text):
                return re.search(rex, text).group(0)
        return ""
    
    # 公司/机构类得分
    def institution_score(self, info:list, ner:list)->list:
        institute = []
        key = "公司|机构|学校|学院|大学|中学|小学|集团|商会|企业"
        char = "局|司"
        for i in range(len(info)):
            if len(info[i]) < 2 or len(info[i]) >= 20:
                institute.append(0)
            elif re.search(key, info[i][-2:]) is not None:
                institute.append(2)
            elif re.search(char, info[i][-1:]) is not None:
                institute.append(2)
            else: institute.append(0)
        for i in range(len(ner)):
            if ner[i][1] == 'ORGANIZATION':
                id = ner[i][2]
                institute[id] = institute[id] + 1.6
        return institute

    # 部门提取
    def department_score(self, info:list, ner:list)->list:
        department = []
        key = "部门|中心|协会"
        char = "部|科|系|院|所|社"
        for i in range(len(info)):
            if len(info[i]) <= 2 or len(info[i]) >= 12:
                department.append(0)
            elif re.search(key, info[i][-2:]) is not None:
                department.append(2)
            elif re.search(char, info[i][-1:]) is not None:
                department.append(2)
            else: department.append(0)
        for i in range(len(ner)):
            if ner[i][1] == 'ORGANIZATION':
                id = ner[i][2]
                department[id] = department[id] + 0.4
        return department

    # 岗位提取
    def job_score(self, info:list)->list:
        segment = HanLP.newSegment().enableOrganizationRecognize(True)
        job = []
        key = "实习|运营|助教|助理|经理|队员|组员|队长|组长|成员|会长|会员|干事|部长|老师"
        char = "岗|师"
        for i in range(len(info)):
            if len(info[i]) < 2 or len(info[i]) >= 12:
                job.append(0)
            elif "实习生" in info[i]:
                job.append(2)
            elif re.search(key, info[i][-2:]) is not None:
                job.append(2)
            elif re.search(char, info[i][-1:]) is not None:
                job.append(1)
            else: job.append(0)
            partition = segment.seg(info[i])
            for k in partition:
                if str(k).endswith("nnd") or str(k).endswith("nnt"):
                    job[i] = 2
                    break  
        return job
    
    # 地点提取
    def loc_extract(self, text:str)->str:
        segment = HanLP.newSegment()
        res = segment.seg(text)

        loc = ""

        for x in res:
            if str(x).endswith("ns"):
                loc = loc + x.word               
        
        if loc == "":
            return None
        else: return loc

    # 行业提取
    def industry_extract(self, text:str, industry_lib=None)->str:
        if industry_lib is None:
            industry_lib = ["互联网","IT","制造业","金融","房地产","建筑","贸易",
                  "零售","物流","教育","传媒","广告",
                  "服务业","市场","销售","人事","财务","行政"]
        assert isinstance(industry_lib, list)
        for i in industry_lib:
            if re.search(i, text):
                return i
        if len(text) >= 4 and len(text) <= 12:
            if re.search("行业[：:\s]", text[:3]):
                return text[3:]
            elif len(text) >= 6:
                if re.search("公司行业[：:\s]", text[:5]):
                    return text[5:]
        else: return None

    # 薪资水平提取
    def extract_money(self, text:str):
        if re.search("工资|薪水|月薪|年薪|日薪|[元￥\$]", text) is None:
            return None
        pattern = "\d+\-\d+元|\$\d+\-\d+|￥\d+-\d+|\d+-\d+|\d+"
        salary = re.search(pattern, text)
        if salary is None:
            return None
        elif "年" in text:
            return salary.group() + "/年"
        elif "月" in text:
            return salary.group() + "/月"
        elif "日" in text or "天" in text:
            return salary.group() + "/天"
        else: return salary.group()
    
    # 提取教育信息
    def education_extract(self, exp:dict):
        info = self.info_process(exp)
        cur_info = {}
        cur_info["school"] = ""
        cur_info["degree"] = ""
        cur_info["major"] = ""
        cur_info["department"] = ""
        cur_info["GPA"] = ""

        for i in info:
            if self.get_school(i) != "" and cur_info["school"] == "":
                cur_info["school"] = self.get_school(i)
            elif self.get_department(i) != "" and cur_info["department"] == "":
                cur_info["department"] = self.get_department(i)
            elif self.get_degree(i) != "" and cur_info["degree"] == "":
                cur_info["degree"] = self.get_degree(i)
            elif self.get_major(i) != "" and cur_info["major"] == "":
                cur_info["major"] = self.get_major(i)
            elif self.get_GPA(i) != "" and cur_info["GPA"] == "":
                cur_info["GPA"] = self.get_GPA(i)

        return cur_info
    
    def correct_eduinfo(self, lexp:list)->list:
        res_list = []
        for i in lexp:
            res_list.append(self.education_extract(i).copy())
        return res_list
    
    # 提取学生主修课程，草稿版
    def courses_extract(self, info:list)->list:
        key = "主修课程|主要课程|已修课程|核心课程|相关课程|课程"
        res = []
        flag = False
        for i, text in enumerate(info):
            if re.search(key, text):
                text = re.sub(key, "", text)
                text = re.sub("：", "", text)
                if len(text) >= 2:
                    res = res + re.split("\s|、|，|；", text)
                i += 1
                while (i < len(info)):
                    text = info[i]
                    if re.search("荣誉|奖项|保研|学分绩|经历", text):
                        break
                    else:
                        if len(text) >= 2:
                            res = res + re.split("\s|、|，|；", text)
                    i += 1
                flag = True
            if flag:
                break
        return res


    # 初步处理工作信息
    def workinfo_extract(self, exp:dict):
        info = self.info_process(exp)
        k = self.ner(info)
        ins = np.array(self.institution_score(info, k))
        dep = np.array(self.department_score(info, k))
        job = np.array(self.job_score(info))
        pos_score = np.concatenate((ins, dep, job)).reshape(3, len(info))
        cur_info = {}
        cur_info["company"] = ""
        cur_info["department"] = ""
        cur_info["job_title"] = ""
        cur_info["location"] = ""
        cur_info["salary"] = ""
        cur_info["industry"] = ""
        length = len(info)

        for i in range(len(k)):
            # 剔除日期、地点、人名
            if k[i][1] == "DATE" or k[i][1] == "TIME" or k[i][1] == "INTEGER":
                id = k[i][2]
                pos_score[:, id] = -1
            if k[i][1] == "LOCATION" or k[i][1] == "PERSON":
                id = k[i][2]
                pos_score[0, id] = pos_score[0, id] - 0.5
                pos_score[1:, id] = -1
        flag1 = flag2 = flag3 = True
        for i in range(len(info)):
            # 薪资、日期、地点等实体，权值矩阵所在列扣分（地点除外）
            # 假设：下列几个字段互斥
            if flag1 and self.extract_money(info[i]) is not None:
                pos_score[:, i] = pos_score[:, i] - 1
                cur_info["salary"] = self.extract_money(info[i])
                flag1 = False
            elif flag2 and self.industry_extract(info[i]) is not None:
                pos_score[:, i] = pos_score[:, i] - 0.5
                cur_info["industry"] = self.industry_extract(info[i])
                flag2 = False
            elif flag3 and self.loc_extract(info[i]) is not None:
                cur_info["location"] = self.loc_extract(info[i])
                flag3 = False
            else:
                continue
        
        if length == 0:
            return pos_score, cur_info, length, info
        # 假设：短语分词情况良好，所有信息出现位置互斥
        if pos_score[0, pos_score[0].argmax()] > 0:
            cur_info["company"] = info[pos_score[0].argmax()]
            pos_score[1:, pos_score[0].argmax()] = -1
        if pos_score[1, pos_score[1].argmax()] > 0:
            cur_info["department"] = info[pos_score[1].argmax()]
            pos_score[2, pos_score[1].argmax()] = -1
        if pos_score[2, pos_score[2].argmax()] > 0:
            cur_info["job_title"] = info[pos_score[2].argmax()]
            pos_score[1, pos_score[2].argmax()] = -1
        return pos_score, cur_info, length, info
    
    # 工作经历提取
    def correct_workinfo(self, lexp:list)->list:
        pos_list = []
        res_list = []
        info_list = []
        min_len = 10

        # 初步提取信息
        for i in lexp:
            pos, res, length, info = self.workinfo_extract(i)
            pos_list.append(pos.copy())
            res_list.append(res.copy())
            info_list.append(info.copy())
            if length < min_len:
                min_len = length
        
        if min_len == 0:
            return res_list
        
        # 更新位置得分
        pos_score = np.zeros((3, min_len))
        for i in pos_list:
            pos_score = pos_score + i[:,0:min_len]
        pos_score = pos_score / len(pos_list) / 2
        
        # 更新依赖pos提取的信息
        for i in range(len(res_list)):
            pos_list[i][:,0:min_len] = pos_list[i][:,0:min_len] + pos_score
            # 假设：短语分词情况良好
            if pos_list[i][0, pos_list[i][0].argmax()] > 0:
                res_list[i]["company"] = info_list[i][pos_list[i][0].argmax()]
            if pos_list[i][1, pos_list[i][1].argmax()] > 0:
                res_list[i]["department"] = info_list[i][pos_list[i][1].argmax()]
            if pos_list[i][2, pos_list[i][2].argmax()] > 0:
                res_list[i]["job_title"] = info_list[i][pos_list[i][2].argmax()]
        return res_list
    
    def name_extract(self, info:list)->list:
        # 项目名，关键词+位置综合考虑
        name = []
        key = "论文|项目|实践|支教|竞赛"
        signals = "“|”|《|》"

        for i in range(len(info)):
            if re.search(key, info[i]):
                t = 2 - 0.2 * i - 0.1 * np.log((len(info[i]) - 20)**2 + 1)
                name.append(t)
            elif re.search(signals, info[i]):
                t = 1.5 - 0.2 * i + 0.1 * np.log((len(info[i]) - 20)**2 + 1)
                name.append(0.8)
            else:
                t = 1 - 0.2 * i + 0.1 * np.log((len(info[i]) - 20)**2 + 1)
                name.append(0)
        return name

    def level_extract(self, text:str)->str:
        # 默认是校级项目
        nation = "国家级|国级|中国[\S]+项目|国家[\S]+项目|中国[\S]+课题|国家[\S]+课题"
        province = "省级|市级|省市级|省[\S]+项目|省[\S]+课题|市[\S]+项目|市[\S]+课题"

        if re.search(nation, text):
            return "国家级"
        elif re.search(province, text):
            return "省市级"
        else: return None
    
    def project_extract(self, exp:dict):
        info = self.info_process(exp)
        k = self.ner(info)
        projects = np.array(self.name_extract(info))
        ins = np.array(self.institution_score(info, k))
        job = np.array(self.job_score(info))
        pos_score = np.concatenate((projects, ins, job)).reshape(3, len(info))
        cur_info = {}
        cur_info["project"] = ""
        cur_info["project_level"] = ""
        cur_info["institute"] = ""
        cur_info["location"] = ""
        cur_info["job_title"] = ""
        length = len(info)

        for i in range(len(k)):
            # 剔除日期、地点、人名
            if k[i][1] == "DATE" or k[i][1] == "TIME" or k[i][1] == "INTEGER":
                id = k[i][2]
                pos_score[:, id] = -1
            if k[i][1] == "LOCATION" or k[i][1] == "PERSON":
                id = k[i][2]
                pos_score[0, id] = pos_score[0, id] - 0.5
                pos_score[1:, id] = -1
        
        flag1 = flag2 = True
        for i in range(len(info)):
            # 日期、地点等实体，权值矩阵所在列扣分（地点除外）
            # 假设：下列几个字段互斥
            if flag1 and self.loc_extract(info[i]) is not None:
                cur_info["location"] = self.loc_extract(info[i])
                flag1 = False
            elif flag2 and self.level_extract(info[i]) is not None:
                cur_info["project_level"] = self.level_extract(info[i])
            else:
                continue
        
        if length == 0:
            return pos_score, cur_info, length, info

        # 假设：短语分词情况良好，所有信息出现位置互斥
        if pos_score[0, pos_score[0].argmax()] > 0:
            cur_info["project"] = info[pos_score[0].argmax()]
        if pos_score[1, pos_score[1].argmax()] > 0:
            cur_info["institute"] = info[pos_score[1].argmax()]
            pos_score[2, pos_score[1].argmax()] = -1
        if pos_score[2, pos_score[2].argmax()] > 0:
            cur_info["job_title"] = info[pos_score[2].argmax()]
            pos_score[1, pos_score[2].argmax()] = -1

        return pos_score, cur_info, length, info
    
        # 工作经历提取
    def correct_projectinfo(self, lexp:list)->list:
        pos_list = []
        res_list = []
        info_list = []
        min_len = 10

        # 初步提取信息
        for i in lexp:
            pos, res, length, info = self.project_extract(i)
            pos_list.append(pos.copy())
            res_list.append(res.copy())
            info_list.append(info.copy())
            if length < min_len:
                min_len = length
        
        if min_len == 0:
            return res_list
        
        # 更新位置得分
        pos_score = np.zeros((3, min_len))
        for i in pos_list:
            pos_score = pos_score + i[:,0:min_len]
        pos_score = pos_score / len(pos_list) / 2
        
        # 更新依赖pos提取的信息
        for i in range(len(res_list)):
            pos_list[i][:,0:min_len] = pos_list[i][:,0:min_len] + pos_score
            # 假设：短语分词情况良好
            if pos_list[i][0, pos_list[i][0].argmax()] > 0:
                res_list[i]["project"] = info_list[i][pos_list[i][0].argmax()]
            if pos_list[i][1, pos_list[i][1].argmax()] > 0:
                res_list[i]["institute"] = info_list[i][pos_list[i][1].argmax()]
            if pos_list[i][2, pos_list[i][2].argmax()] > 0:
                res_list[i]["job_title"] = info_list[i][pos_list[i][2].argmax()]
        return res_list 

# 技能提取
def extract_skill(text:str)->list:
    '''Extract professional words from texts'''
    segment = HanLP.newSegment().enableOrganizationRecognize(True)
    res = segment.seg(text)

    expected_n = ['/g', '/gb', '/gbc', '/gc', '/gg', '/gi', '/gm', '/gp','nx']
    l = [x.word for x in res for y in expected_n if str(x).endswith(y)]
    return l 