from setuptools import setup, find_packages

# 解析 requirements.txt 文件，获取依赖包列表
def parse_requirements(filename):
    # 逐行读取文件，去除首尾空白字符
    lineiter = (line.strip() for line in open(filename))
    # 过滤掉空行和以 # 开头的注释行
    return [line for line in lineiter if line and not line.startswith("#")]

# 读取依赖包列表
reqs = parse_requirements('requirements.txt')

# 包的版本号
VERSION = '3.0.1'
# 简短描述
DESCRIPTION = 'Only Train Once (OTO): Automatic One-Shot General DNN Training and Compression Framework'
# 详细描述
LONG_DESCRIPTION = 'Only Train Once (OTO): Automatic One-Shot General DNN Training and Compression Framework'

# 调用 setup() 配置包信息
setup(
    name="only_train_once",  # 包名
    version=VERSION,         # 版本号
    description=DESCRIPTION, # 简短描述
    long_description=LONG_DESCRIPTION, # 详细描述
    author="Tianyi Chen",    # 作者
    author_email="tiachen@microsoft.com", # 作者邮箱
    license='MIT',           # 许可证
    packages=find_packages(),# 自动查找所有子包
    install_requires=reqs,   # 安装依赖
    url="https://github.com/tianyic/only_train_once", # 项目主页
    keywords='automatic, one-shot, structure pruning, sparse optimization', # 关键词
    classifiers= [           # 分类信息
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)