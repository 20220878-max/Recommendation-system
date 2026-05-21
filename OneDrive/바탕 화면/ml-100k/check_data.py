# %%
import os

print("===== 파일 목록 =====")
print(os.listdir("."))


# %%
def show_rating_file(filename, n=10):
    print(f"\n===== {filename} 미리보기 =====")
    print("user_id\titem_id\trating\ttimestamp")
    print("-" * 45)

    with open(filename, "r", encoding="utf-8") as f:
        for i in range(n):
            line = f.readline().strip()
            user_id, item_id, rating, timestamp = line.split("\t")
            print(user_id, item_id, rating, timestamp, sep="\t")


def summarize_rating_file(filename):
    users = set()
    items = set()
    ratings = set()
    total_rating = 0
    count = 0

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            user_id, item_id, rating, timestamp = line.strip().split("\t")

            users.add(user_id)
            items.add(item_id)
            ratings.add(int(rating))
            total_rating += int(rating)
            count += 1

    print(f"\n===== {filename} 요약 =====")
    print("평점 개수:", count)
    print("사용자 수:", len(users))
    print("영화 수:", len(items))
    print("평점 종류:", sorted(ratings))
    print("평균 평점:", total_rating / count)


# %%
show_rating_file("u.data")
summarize_rating_file("u.data")


# %%
show_rating_file("ua.base")
summarize_rating_file("ua.base")


# %%
print("\n===== u.item 영화 정보 미리보기 =====")

with open("u.item", "r", encoding="latin-1") as f:
    for i in range(10):
        line = f.readline().strip()
        parts = line.split("|")

        movie_id = parts[0]
        title = parts[1]
        release_date = parts[2]

        print("영화 ID:", movie_id)
        print("제목:", title)
        print("개봉일:", release_date)
        print("-" * 40)


# %%
print("\n===== u.user 사용자 정보 미리보기 =====")
print("user_id | age | gender | occupation | zip_code")
print("-" * 55)

with open("u.user", "r", encoding="utf-8") as f:
    for i in range(10):
        line = f.readline().strip()
        user_id, age, gender, occupation, zip_code = line.split("|")
        print(user_id, age, gender, occupation, zip_code, sep=" | ")


# %%
print("\n===== u.genre 장르 목록 =====")

with open("u.genre", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())