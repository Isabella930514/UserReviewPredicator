from UserReviewPredicator import URP


def main():
    model = URP('review_Kindle_Store.csv', 0, 4, 5, 5, 10)
    result = model.get_predication_result(ratio=0.2)
    print(result)


if __name__ == "__main__":
    main()
