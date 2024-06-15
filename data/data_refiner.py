import json

file_name = input('File name? ')

with open(f'{file_name}.json', 'r') as file:
    jsonData = json.load(file)

real_data = {}
real_posts = []
total_posts = []

words = {}


for user, posts in jsonData.items():
    total_posts.extend(posts)

    posts = [post for post in posts if len(post[1][1].split()) >= 10 ]

    # posts = [post for post in posts if post[0] != file_name]

    for post in posts:
        if post[0] == 'MadeMeSmile': raise Exception
        for word in post[1][1].split():
            words[word] = words.get(word, 0) + 1

    if len(posts) != 0:
        real_data[user] = posts
        real_posts.extend(posts)
    
print('Total users:', len(real_data))
print('Total posts:', len(total_posts))
print(f'Posts not in r/{file_name}:', len(real_posts))

print(f'Most common words used by r/{file_name} users: ', end='')
print(sorted(words.items(), key=lambda w: w[1], reverse=True)[:5])


with open(f'{file_name}-posts.json', 'w') as outputFile:
    json.dump(real_data, outputFile, indent=4)