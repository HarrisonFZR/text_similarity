# -*- coding: utf-8 -*
# import necessary packages.
import pandas as pd
import numpy as np
import zipfile
import re
import datetime
#git pull and git fetch locally
# define a function so that we can get yesterday's date.

df_ios = pd.read_csv('UTF-8\'\'HD_iOS_Daily_Auto_Final_r120211102.csv')
df_android = pd.read_csv('UTF-8\'\'HD_And_Auto_Final_r120211102.csv')
df_web = pd.read_csv('HD_Web_Daily_Auto_Final20211102.csv')
      

# 2. calculate sum of pv and uv.
df_ios.rename(columns={'Article Name/ID (prop4)':'Article Name', 'Site Section (evar2)':'Site Section'},inplace=True)
df_android.rename(columns={'Article Name (evar4)':'Article Name', 'Site Section (evar2)':'Site Section'},inplace = True)
df_web.rename(columns={'Article Name (evar5)':'Article Name', 'Site Section (evar3)':'Site Section'},inplace = True)

df_ios = df_ios.fillna(0)
df_android = df_android.fillna(0)
df_web = df_web.fillna(0)

df_web.drop(['Page Name (prop1)','Page URL (evar2)', 'Unnamed: 0'],axis=1,inplace=True)
df_ios.drop(['Article ID (prop22)'], axis=1, inplace = True)
df_android.drop(['Article ID (prop22)'], axis=1, inplace = True)

# 2.1 Article name cleaning part.
def filter_tags(htmlstr):
  re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I)
  re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)
  re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)
  re_br=re.compile('<br\s*?/?>')
  re_h=re.compile('</?\w+[^>]*>')
  re_comment=re.compile('<!--[^>]*-->')
  s=re_cdata.sub('',htmlstr)
  s=re_script.sub('',s)
  s=re_style.sub('',s)
  s=re_br.sub('\n',s)
  s=re_h.sub('',s)
  s=re_comment.sub('',s)
  blank_line=re.compile('\n+')
  s=blank_line.sub('\n',s)
  s=replaceCharEntity(s)
  return s

def replaceCharEntity(htmlstr):
  CHAR_ENTITIES={'nbsp':' ','160':' ',
        'lt':'<','60':'<',
        'gt':'>','62':'>',
        'amp':'&','38':'&',
        'quot':'"','34':'"', '039':'\'', 
        'mdash':'-', 'ensp':' ',
        'hellip':'…','middot':'·', #'hellip':'…','middot':'·','hellip':'...','middot':'.'
        'deg':'°', 'ndash':'–',
        'bull':'•', '-':'—'}
   
  re_charEntity = re.compile(r'&#?(?P<name>\w+);')
  sz = re_charEntity.search(htmlstr)
    
  while sz:
    entity = sz.group()
    key = sz.group('name')
    try:
      htmlstr = re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
      sz = re_charEntity.search(htmlstr)
    except KeyError:
      htmlstr = re_charEntity.sub('',htmlstr,1)
      sz = re_charEntity.search(htmlstr)
  return (htmlstr)

def replace(s,re_exp,repl_string):
  return (re_exp.sub(repl_string,s))

if __name__=='__main__':

  for i in range(len(df_ios['Article Name'])):
    if df_ios['Article Name'][i] == 0:
      continue
    else:
      df_ios['Article Name'][i] = filter_tags(df_ios['Article Name'][i])
      
  for i in range(len(df_android['Article Name'])):
    if df_android['Article Name'][i] == 0:
      continue
    else:
      df_android['Article Name'][i] = filter_tags(df_android['Article Name'][i])
      
  for i in range(len(df_web['Article Name'])):
    if df_web['Article Name'][i] == 0:
      continue
    else:
      df_web['Article Name'][i] = filter_tags(df_web['Article Name'][i])
    
  df_ios['Article Name'] = df_ios['Article Name'].str.strip()
  df_android['Article Name'] = df_android['Article Name'].str.strip()
  df_web['Article Name'] = df_web['Article Name'].str.strip()

#2.2 merge all datasets
df = pd.concat([df_ios, df_android, df_web])
df_sum1 = df.groupby(['Article Name', 'Site Section', 'Date']).sum()
df_sum1 = df_sum1.fillna(0)
df_sum1.reset_index(inplace = True, drop = False)
df_sum1.rename(columns={'Page Views':'Page Views Sum', 'Unique Visitors':'Unique Visitors Sum'},inplace=True)


# 3. merge the SUM file with the ios, android and web file.
df_ios = df_ios.groupby(['Article Name', 'Site Section', 'Date']).sum()
df_ios.reset_index(inplace = True, drop = False)
df_ios = df_ios.fillna(0)

df_sum2 = pd.merge(df_sum1,df_ios,on=['Article Name','Site Section', 'Date'], how = 'outer')
df_sum2.rename(columns={'Page Views':'Page Views ios', 'Unique Visitors':'Unique Visitors ios'},inplace=True)
df_sum2 = df_sum2.fillna(0)

df_android = df_android.groupby(['Article Name', 'Site Section', 'Date']).sum()
df_android.reset_index(inplace = True, drop = False)
df_android = df_android.fillna(0)

df_sum3 = pd.merge(df_sum2,df_android,on=['Article Name','Site Section', 'Date'], how = 'outer')
df_sum3.rename(columns={'Page Views':'Page Views android', 'Unique Visitors':'Unique Visitors android'},inplace=True)
df_sum3 = df_sum3.fillna(0)

df_web = df_web.groupby(['Article Name', 'Site Section']).sum()
df_web.reset_index(inplace = True, drop = False)
df_web = df_web.fillna(0)

df_sum4 = pd.merge(df_sum3,df_web,on=['Article Name','Site Section'], how = 'outer')
df_sum4.rename(columns={'Page Views':'Page Views Web', 'Unique Visitors':'Unique Visitors Web'},inplace=True)
df_sum4 = df_sum4.fillna(0)

#4. add up pv_ios and pv_android
df_sum_pv_app=[]
for i in range(len(df_sum4)):
      df_app_pv = df_sum4['Page Views ios'][i]+df_sum4['Page Views android'][i]
      df_sum_pv_app.append(df_app_pv)
df_sum4['Page Views App'] = df_sum_pv_app

df_sum_uv_app=[]
for i in range(len(df_sum4)):
      df_app_uv = df_sum4['Unique Visitors ios'][i]+df_sum4['Unique Visitors android'][i]
      df_sum_uv_app.append(df_app_uv)
df_sum4['Unique Visitors App'] = df_sum_uv_app

# 5. remove ios and android
df_sum4.drop(['Page Views ios', 'Page Views android', 'Unique Visitors ios', 'Unique Visitors android'], axis=1, inplace = True)

# 6. remove rows containing useless site sections
df_site = pd.read_csv('Site.csv')
df_site_N = list(df_site.loc[df_site['Y/N']=='N']['Site'])
df_sum4_site = df_sum4[~df_sum4['Site Section'].isin(df_site_N)]
df_sum5 = df_sum4_site
df_sum5.reset_index(inplace = True, drop = True)

# 7. adding all together, extracting all Chinese char in 'Article Name' as key to group by articles.

# 7.1 extract all Chinese Char and join them up. The Chinese char will be used as a key to identify a article.
ch = []
pattern="[\u4e00-\u9fa5]+" 
regex = re.compile(pattern)
for i in range(len(df_sum5)):
    result =  regex.findall(df_sum5['Article Name'][i])
    ch.append(result)
    
df_sum5['key'] = ch

for i in range(len(df_sum5)):
    df_sum5['key'][i] = ''.join(df_sum5['key'][i])

# 7.2 As one article with two different names(characters are the same, just with some different punctuations), so just use one name as the article name.  
df_sum6 = df_sum5.groupby(['Date', 'key'],as_index=False).agg(lambda x : x.str.cat(sep='//// ') if x.dtype == 'object' else x.sum())
df_sum8 = df_sum6[~df_sum6['key'].isin([''])]
df_sum8.reset_index(inplace = True, drop = False)

one_name = []
for i in range(len(df_sum8)):
    one = df_sum8['Article Name'][i].split('////')[0]
    one_name.append(one)

df_sum8['Article Name'] = one_name


# text similarity
#Create a key-key binary matrix
key1 = df_sum8.key.unique()
key2 = df_sum8.key.unique()

keyMatrix = pd.DataFrame(columns=key1, index=key2)
keyMatrix.head(2)



""" 
cosine similarity
"""
def get_word_vector(s1, s2):
    """
    :param s1: 字符串1
    :param s2: 字符串2
    :return: 返回字符串切分后的向量
    """
    # 字符串中文按字分，英文按单词，数字按空格
    regEx = re.compile('[\\W]*')
    res = re.compile(r"([\u4e00-\u9fa5])")
    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)
    # print(str1_list)
    p2 = regEx.split(s2.lower())
    str2_list = []
    for str in p2:
        if res.split(str) == None:
            str2_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str2_list.append(ch)
    # print(str2_list)
    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
    list_word2 = [w for w in str2_list if len(w.strip()) > 0]  # 去掉为空的字符
    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))
    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 输出向量
    return word_vector1, word_vector2
  

def cos_dist(vec1, vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return dist1
  
for i in range(len(keyMatrix)):
    for j in range(len(keyMatrix.columns)):
        k1 = keyMatrix.index[i]
        k2 = keyMatrix.columns[j]
        v1, v2 = get_word_vector(k1, k2)
        cos_sim = cos_dist(v1,v2)
        keyMatrix.at[k1, k2] = cos_sim
        
''' a = []
for i in range(len(keyMatrix)):
    for j in range(len(keyMatrix.columns)):
        k1 = keyMatrix.index[i]
        k2 = keyMatrix.columns[j]
        if 0.8 <= keyMatrix.at[k1, k2] < 0.99999999:
            a.append([k1, k2]) '''

keyMatrix.to_csv('keyMatrix.csv', encoding = 'utf_8_sig')

