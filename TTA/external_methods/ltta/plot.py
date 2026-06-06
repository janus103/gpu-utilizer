import matplotlib.pyplot as plt

def draw_plot(f_name, base, lst, colors):
    # 임의의 데이터 포인트
    x = []
    #print('Length of List ',len(lst))
    for idx in range(len(lst)):
        x.append(idx+1)
    #y = [2, 3, -1, 5, 7]
    y = lst

    # 기준선 값
    baseline_value = base

    
    for xi, yi, color in zip(x, y, colors):
        if color == 0:
            # color가 0인 경우 파란색으로 표시
            plt.stem([xi], [yi], 'r', markerfmt='r.', basefmt='blue', bottom=baseline_value)
        elif color == 1:
            # color가 1인 경우 빨간색으로 표시
            plt.stem([xi], [yi], 'g', markerfmt='g.', basefmt='blue', bottom=baseline_value)
    
    
    #plt.stem(x, y, 'b', markerfmt='bo', basefmt='orange', bottom=baseline_value, use_line_collection=True)

    # 축 레이블 추가
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.savefig('sample/'+ f_name +'.png', format='png', dpi=100)
    
    plt.close()
    
    
def plot_three_lists_two_groups(filename, list1, list2, list3):
    len1, len2, len3 = len(list1), len(list2), len(list3)
    if not (len1 == len2 == len3):
        raise ValueError("All three lists must have the same length. {} / {} / {} ".format(len1 , len2 , len3))
    list4 = list()
    for i in range(len1):
        list4.append(list2[i] - list1[i])
    
    x_positions = range(len(list1))

    fig, ax1 = plt.subplots()

    # 첫 번째 막대 그래프 (왼쪽 y축)
    ax1.bar(x_positions, list1, width=0.4, label='True', align='center')
    ax1.bar(x_positions, list4, width=0.4, label='False', align='edge', alpha=0.5)

    # 두 번째 y축 설정
    ax2 = ax1.twinx()
    
    # 꺾은선 그래프 (오른쪽 y축)
    ax2.plot(x_positions, list3, label='Accuracy', color='red', marker='o')
    ax2.set_ylim(-100, 100)
    
    # 레이블 및 타이틀 설정
    ax1.set_xlabel('True & False')
    ax1.set_ylabel('Prob (Mean)', color='blue')
    ax2.set_ylabel('Accuracy', color='red')
    ax1.set_title('Bar and Line Graph with Two Y-Axes')
    
    # 범례 설정
    #ax1.legend(loc='upper left')
    #ax2.legend(loc='upper right')
    
    # 범례 설정 - 그래프 오른쪽 바깥에 위치
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 그래프 출력하기 전에 tight_layout()을 호출하여 오버랩을 방지
    plt.tight_layout()

    # 그래프 보여주기
    plt.savefig('sample/'+filename, format='png', dpi=100)
    
    plt.close()

# # 예제 리스트
# list1 = [5, 10, 15, 20, 25]
# list2 = [3, 8, 12, 18, 22]
# list3 = [2, 5, 8, 12, 28]

# # 함수 실행 및 그래프 출력
# plot_three_lists_two_groups(list1, list2, list3)

