from os import listdir
from os.path import isfile, join
import json
from random import shuffle

files = [f for f in listdir('.') if isfile(join('.', f))]

post_files = [f for f in files if f[-11:] == '-posts.json']

total_posts = []

for post_file in post_files:
    with open(post_file, 'r') as file:
        users = json.load(file)
    
    subreddit_name = post_file[:-11]
    
    for user, posts in users.items():
        for post in posts:
            post = {
                'text': post[1][1],
                'label': 1 if subreddit_name == 'depression' else 0
            }

            total_posts.append(post)

shuffle(total_posts)

with open('total_posts.json', 'w') as outputFile:
    json.dump(total_posts, outputFile, indent=4)