from fire import Fire

import bbh_dev

def main(task_name: str, **kwargs):
    task_map = dict(
        bbh=bbh_dev.main,
    )

    if task_name == "all":
        results = {}
        for name, task_fn in task_map.items():
            score = task_fn(**kwargs)
            results[name] = score
    else:
        task_fn = task_map.get(task_name)
        if task_fn is None:
            raise ValueError(f"{task_name}. Choose from {list(task_map.keys())}")
        score = task_fn(**kwargs)
        results = {task_name: score}

    results = {name: round(score * 100, 2) for name, score in results.items()}
    print(results)
    return results

if __name__ == "__main__":
    Fire(main)
