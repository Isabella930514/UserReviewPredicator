from UserReviewPredicator import URP


def main():
    # filename, reviewer_ID_idx, review_text_idx, review_rate_idx, max_user = None, epoch_no = None, learning_rate, dropout_rate
    model = URP('data.csv', 0, 4, 5, 10, 15, 0.0001, 0.3)
    # split train and test data with 80% and 20%
    result = model.get_predication_result(ratio=0.2)
    print(f'apf:{result[0]}, mae:{result[1]}, mse:{result[2]}, rmse:{result[3]}')


if __name__ == "__main__":
    main()
