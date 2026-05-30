import os
import argparse#解析模块，处理用户传入的脚本参数
import joblib
import pandas as pd

from utils import train_two_stage_model#从自定义工具模块导入两阶段模型训练函数


def read_data(file_path):
    #自动读取 csv / xlsx / xls 数据文件
    file_path_lower = file_path.lower()
    #文件格式识别
    if file_path_lower.endswith(".csv"):
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk")

    elif file_path_lower.endswith(".xlsx") or file_path_lower.endswith(".xls"):
        df = pd.read_excel(file_path)

    else:
        raise ValueError("只支持 csv、xlsx、xls 格式的数据文件")

    # 去掉列名前后的空格
    df.columns = df.columns.str.strip()

    return df#返回处理后的DataFrame


def main():
    parser = argparse.ArgumentParser()#创建命令行参数解析器实例

    parser.add_argument(
        "--file",
        type=str,
        required=True,#标记为必填参数
        help="训练数据文件路径，支持 csv、xlsx、xls"
    )

    parser.add_argument(
        "--target",
        type=str,
        default="x1",#默认目标列
        help="目标列名，例如 x1 或 y1"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",#默认模型保存目录为当前目录下的models文件
        help="模型保存目录"
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("当前工作目录：")
    print(os.getcwd())
    print("=" * 60)

    print("开始读取数据文件：")
    print(args.file)

    df = read_data(args.file)#调用函数读取数据

    print("数据读取成功！")
    print("数据维度：", df.shape)
    print("数据列名：")
    print(df.columns.tolist())#确认目标列是否存在
    print("数据前5行：")
    print(df.head())#检查数据格式是否正确

    print("=" * 60)
    print("开始训练模型")
    print("=" * 60)

    bundle, result_df = train_two_stage_model(
        df,
        target_col=args.target
    )#调用两阶段模型训练函数，返回模型包和预测结果

    model_path = os.path.join(
        args.save_dir,
        f"two_stage_{args.target}.pkl"
    )

    result_path = os.path.join(
        args.save_dir,
        f"prediction_results_{args.target}.csv"
    )

    joblib.dump(bundle, model_path)

    result_df.to_csv(
        result_path,
        index=False,#不保存索引
        encoding="utf-8-sig"
    )

    print("=" * 60)
    print("模型训练完成！")
    print("模型保存路径：", model_path)
    print("训练预测结果保存路径：", result_path)

    print("\n训练集指标：")
    print(bundle["metrics"]["train"])

    print("\n测试集指标：")
    print(bundle["metrics"]["test"])
    print("=" * 60)


if __name__ == "__main__":
    main()#当脚本直接执行时运行main函数
