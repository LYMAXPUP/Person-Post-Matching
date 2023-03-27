import re
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.converter import PDFPageAggregator
from collections import defaultdict
import pandas as pd
import chardet


text_dict = {}
text_prop_dict = {}  # text的属性，这个是一个dict
pdf_to_text_list = []

def write_txt_file(text, output_txtpath):
    with open(output_txtpath,"w", encoding='utf-8') as txt:
        for i in range(len(text)):
            s = str(text[i])
            txt.write(s)   # 自带文件关闭功能，不需要再写 f.close()


def is_chinese(string):   # 不能光是中文，这个应该是判断是不是一堆空格的
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def is_text(text):   
    """
    检查整个字符串是否有空格
    :param string: 需要检查的字符串
    :return: bool
    """
    if text != " \n":
        return True
    else:
        False


def is_leftright(df):
    if df["x0"][df["x0"]>190].count() > 20:
        return True
    else:
        False


def del_duplicate(text_list):  # list -> list   这边清理之后 分模块再处理的话只需要 del text_list[0]第一个标题就可以了
    text = []
    j = ""
    for i in text_list:
        # 去除空行
        if re.sub("\s", "", i) == "":
            continue
        elif re.sub("\s", "", i) == j:
            continue
        else:
            j = re.sub("\s", "", i)
            line_elem = re.split("\s[\s]+|\n", i)
            for elem in line_elem:
                if elem != "" and elem != " ":
                    text.append(elem)
    text = sorted(set(text),key=text.index)  # 把重复的元素删除，且保留原有的顺序
    return text

# 阅读存入字典中
# 这个用的是 pdfminer 来解析，但是他没办法保证pdf原本的格式
def read_pdf_miner(file):
    """
    This function takes the file object, read the file content and store it into a dictionary for processing

    :param fileObj: File object for reading the file
    :return: None
    """
    parser = PDFParser(file)  # 用文件对象来创建一个pdf文档分析器，PDFParser从文件中提取数据
    document = PDFDocument(parser)  # 创建一个PDF文档 PDFDocument保存数据，存储文档数据结构到内存中 
    if not document.is_extractable:   # 检测文件是否提供txt转换，不提供就忽视
        raise PDFTextExtractionNotAllowed
    rsrcmgr = PDFResourceManager()    # PDFResourceManager：pdf 共享资源管理器,用于存储共享资源，如字体或图像。
    device = PDFDevice(rsrcmgr)       # 把解析到的内容转化为你需要的东西
    
    # BEGIN LAYOUT ANALYSIS
    laparams = LAParams()     # 加入需要解析的参数   
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)  # 创建一个PDF设备对象
    interpreter = PDFPageInterpreter(rsrcmgr, device)    # 创建一个PDF解释器对象，解析page内容
    page_number = len([page for page in PDFPage.create_pages(document)])  # 循环遍历列表，每次处理一个page的内容
    if page_number > 1:
        text_list = []   # 这个list是用来装全部的参数的  
        for page in PDFPage.create_pages(document):
            page_dict =  defaultdict(list)
            interpreter.process_page(page)
            layout = device.get_result() 
            page_list = []
            for lt_obj in layout:         # 遍历每一页的每一个元素     
                if isinstance(lt_obj, LTTextBoxHorizontal):
                    if is_text(lt_obj.get_text()) is True:
                        page_dict["y0"].append(lt_obj.bbox[1])
                        page_dict["text"].append(lt_obj.get_text())
                        page_df = pd.DataFrame(page_dict).sort_values("y0", ascending=False) # 依照y0排序
                        page_list = page_df["text"].tolist()
                    else:
                        pass
            text_list.extend(page_list)   # 多维的list合并为一维
    else:
        for page in PDFPage.create_pages(document):
            page_dict =  defaultdict(list)    # 建立一个空的dict，以dict存储
            # page_prop_dict =  {}
            interpreter.process_page(page)
            layout = device.get_result()
            # id = 0
            for lt_obj in layout:   
                if isinstance(lt_obj, LTTextBoxHorizontal):
                    if is_text(lt_obj.get_text()) is True:
                        page_dict["x0"].append(lt_obj.bbox[0])   # 利用 x0筛选左右和上下的格式
                        page_dict["y0"].append(lt_obj.bbox[1])   # 利用 y0来排序
                        page_dict["text"].append(lt_obj.get_text())     # 保存每个模块的字段
                        # page_prop_dict[id] = lt_obj
                    else:
                        pass
                    # id += 1
            page_df = pd.DataFrame(page_dict)
            if is_leftright(page_df):          # 左右格式的情况
                df_left = page_df[page_df['x0']<=190].sort_values("y0", ascending=False)
                df_right = page_df[page_df['x0']>190].sort_values("y0", ascending=False)
                page_df = pd.concat([df_left, df_right], axis=0, ignore_index=True)
            else:
                page_df = page_df.sort_values("y0", ascending=False)
            text_list = page_df["text"].tolist()
    return text_list

def rebytes(text_list):
    text = []
    for i in text_list:
        if isinstance(i,bytes):
            encoding = chardet.detect(i)['encoding']
            i = i.decode(encoding)
        text.append(i)
    return text

def transform_pdf(file):
    file = open(file, 'rb') # 要用二进制
    text_list = read_pdf_miner(file)
    return text_list