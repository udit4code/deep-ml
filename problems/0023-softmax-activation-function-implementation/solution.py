import math



def softmax(scores: list[float]) -> list[float]:
	# Your code here
    probabilities = [ ]
    all_exp_vals = [ ]
    total = 0
    for score in scores:
        exp_val = math.exp(score)
        all_exp_vals.append(exp_val)
        total += exp_val
    # print(f"total : {total}")
    for exp_val in all_exp_vals:
        temp = exp_val / total
        probabilities.append(round(temp, 4))
	return probabilities