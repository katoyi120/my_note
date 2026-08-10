# 机器学习笔记

基于 scikit-learn 的机器学习学习笔记。本仓库使用 **Jupyter Book** 将
`MachineLearning/sciikit_learn/` 目录下的 Jupyter Notebook 构建成在线网页，
并通过 GitHub Actions 自动部署到 **GitHub Pages**。

## 在线文档

每次向 `main` 分支推送代码后，GitHub Actions 会自动构建并发布，访问地址：

<https://katoyi120.github.io/my_note/>

> 首次部署前，需要在 GitHub 仓库的
> **Settings → Pages → Build and deployment → Source** 中选择
> **GitHub Actions**，然后推送一次即可自动生成。

## 项目结构

```text
.
├── myst.yml                          # Jupyter Book 站点配置
├── index.md                          # 网站首页
├── .github/workflows/deploy-book.yml # 自动构建并部署到 GitHub Pages
└── MachineLearning/sciikit_learn/    # 笔记内容（.ipynb）
    └── data/                         # 数据文件（不参与网页构建）
```

## 添加新笔记

把新的 `.ipynb` 文件放进 `MachineLearning/sciikit_learn/` 目录并推送到
GitHub 即可，网页会自动更新，无需修改任何配置。

注意两点：

1. 建议使用英文或拼音文件名（例如 `svm.ipynb`）。纯中文文件名在生成
   网页 URL 时会被转换成空字符串，导致页面路径冲突，无法正常访问。
2. Notebook 中请保留运行结果（输出），这样构建时直接展示已有输出，
   不会在服务器上重新执行代码，也无需安装机器学习依赖。

## 本地预览

需要 Python 3.10+（构建时还需要 Node.js 18+，Jupyter Book 2 会自行调用）：

```bash
pip install jupyter-book
jupyter-book build --html
jupyter-book start .
```

然后在浏览器打开 <http://localhost:3000>。

## 工作原理

```mermaid
flowchart LR
    A[推送代码到 GitHub main 分支] --> B[GitHub Actions 安装 Jupyter Book]
    B --> C[jupyter-book build 生成静态网页]
    C --> D[部署到 GitHub Pages]
```

Notebook 中的代码不会在构建时执行，页面直接渲染 Notebook 中已保存的
代码和输出；如需重新执行所有 notebook，可运行
`jupyter-book build --html --execute`。
