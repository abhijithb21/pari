from bs4 import BeautifulSoup
from urlparse import urlparse
import requests
import re
from functools import reduce

class ReadTimeEstimation():
    def get_video_ids(self,document):
        soup = BeautifulSoup(document, 'html.parser')
        embed_list = soup.findAll('embed')
        video_ids = []
        if len(embed_list) >= 1:
            for embed in embed_list:
                url = embed.attrs.get("url")
                video_ids.append(urlparse(url).path)
        return video_ids

    def get_video_durations(self, video_id_list):
        # AIzaSyBqO3eZOPzkIEpiiHQOC - OjxDGaBvK64kA
        if len(video_id_list) > 0:
            videos_durations = []
            videos_durations_dict = {"H":0,"M":0,"S":0}
            for video_id in video_id_list:
                id = video_id.replace('/','')
                url = "https://www.googleapis.com/youtube/v3/videos?id="+id + "&key=<API Key>&part=snippet,contentDetails"
                response = requests.get(url)
                if (response.ok):
                    json_response = response.json()
                    non_formated_duration = json_response.items()[0][1][0]["contentDetails"]["duration"]
                    time = non_formated_duration[2:len(non_formated_duration)]
                    time_array = list(map(int, re.split('[HMS]', "2M1S")[:-1]))
                    videos_durations.append(time_array)

            return videos_durations


    def extract_text(self,document):
        soup = BeautifulSoup(document, 'html.parser')
        text = soup.findAll(text=True)
        return text

    def count_words_in_text(self,text_list,word_length):
        total_words = 0
        for current_text in text_list:
            total_words += len(current_text) / word_length
        return total_words

    def get_read_time_for_text(self,content):
        WPM = 200
        WORD_LENGTH = 5
        text_list = self.extract_text(content)
        video_ids = self.get_video_ids(content)
        video_duration = self.get_video_durations(video_ids)
        total_words = self.count_words_in_text(text_list, WORD_LENGTH)
        return total_words / WPM

    def add_time_arrys(self,first_time_array,second_time_array):
        if len(first_time_array) == len(second_time_array) == 2:
            return [first_time_array[0]+second_time_array[0]]

    def compain_time(self, video_duration_list,text_readtime):
        compained_video_time = reduce((lambda x, y: add(x, y)), video_duration_list)


    def get_read_time_for_article(self,content):
        text_read_time = self.get_read_time_for_text(content)
        # video_id_list = self.get_video_ids(content)
        # video_durations = self.get_video_durations(video_id_list)
        # self.compain_time(video_durations,text_read_time)
        return text_read_time


