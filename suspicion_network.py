
import pandas

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 13:50:58 2016

@author: vk14489
"""
import pandas as pd
import os
import networkx as nx

# From one to one node data, following function generates nodes and edges

def tweetCount (author_list , retweet_list , original_list):
    """

    :param author_list: list of retweeting authors
    :param retweet_list: flag of retweet or not
    :param original_list: list of original author (same as author_list value if not retweet)
    :return:
    """

    indices = [x[0] for x in enumerate(retweet_list) if x[1] == 1]
    a_list=[x[1] for x in enumerate(author_list) if x[0] in indices]
    o_list=[x[1] for x in enumerate(original_list) if x[0] in indices]
    dict_graph={}
    for i in xrange(len(a_list)):
        dict_graph[(a_list[i],o_list[i])] = dict_graph.get((a_list[i],o_list[i]),0) + 1

    df_out = pd.DataFrame()
    df_out["Source"]=[x[0] for x in dict_graph.keys()]
    df_out["Target"]=[x[1] for x in dict_graph.keys()]
    df_out["Weight"]=dict_graph.values()

    nodes = list(set(df_out.Source).union(set(df_out.Target)))
    df_out2 = pandas.DataFrame({'Id':nodes,'Label':nodes})
    return df_out,df_out2


if __name__ == '__main__':

    import pandas
    df = pandas.read_csv("mail_mapping.csv")

    df_out1,df_out2 = tweetCount(list(df.screen_name),list(df.retweet_or_not),list(df.screen_name_o))
    df_out1.to_csv("edges_enron.csv")
    df_out2.to_csv("nodes_enron.csv")
