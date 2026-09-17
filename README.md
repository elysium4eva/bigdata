# 大数据分析［2026 国际商务］课程主页

这是深圳大学《大数据分析［2026 国际商务］》（Big Data Analytics）课程主页的完整站点文件。
推送到 GitHub 仓库 `bigdata` 并开启 Pages 后，访问地址为：

**https://elysium4eva.github.io/bigdata/**

## 站点结构

```
website/                     ← 把这个目录的内容放到仓库 bigdata 的根目录
├── index.html               主页：课程信息 / 教学进度（第 3–14 周）/ 章节与课件 / 考核 / 参考 / 政策
├── syllabus.html            课程大纲全文（可在线打印或另存 PDF）
├── lecture-01.html          第一章课件页（含要点、下载按钮与在线预览）
├── assets/
│   ├── css/style.css        深大配色样式（荔枝红 #8A0C3C + 金 #C9A227）
│   └── img/                 校徽 szu-logo.jpg、封面背景 background.jpg
├── files/                   课件与大纲文件（供下载）
│   ├── L01-introduction-2026.pdf                        第一章课件（2026 版，82 页，2.3 MB）
│   ├── L01-introduction-converted.pdf                   第一章课件（原版逐页转换，78 页，34.7 MB）
│   ├── L06-logistic-regression-linear-classifier.pdf    第六章课件（Beamer 版，80 页，16.8 MB）
│   ├── Syllabus-Big-Data-Analytics-2026.pdf             课程大纲 PDF
│   └── Syllabus-Big-Data-Analytics-2026.docx            课程大纲 Word
├── .nojekyll                关闭 Jekyll 处理（保证 files/ 等目录原样发布）
└── publish.ps1              一键初始化仓库并推送（需先安装 Git 并登录 GitHub）
```

## 发布步骤（三选一）

### 方式 A：一键脚本（推荐）

本目录的 git 仓库**已初始化并暂存全部文件**，remote 已指向
`https://github.com/elysium4eva/bigdata.git`。你需要做的只有两步：

1. 在 GitHub 上新建仓库 **`bigdata`**（Public，不要勾选初始化 README）。
2. 在本目录打开 PowerShell 运行：

```powershell
.\publish.ps1          # 提交并推送（首次会要求 GitHub 登录/授权）
```

3. 打开仓库的 **Settings → Pages**，Source 选择 **Deploy from a branch**，
   Branch 选 **main**、目录选 **/(root)**，保存。
4. 约 1 分钟后访问：https://elysium4eva.github.io/bigdata/

> 若仓库地址不同，用 `.\publish.ps1 -Remote "https://github.com/<用户名>/<仓库>.git"` 覆盖。
> 脚本会自动补一个仓库级 git 身份（`<用户名>@users.noreply.github.com`）；
> 想让提交归属到你的账号，可先设置成 GitHub 提供的 noreply 邮箱：
> `git config user.email "你的ID+elysium4eva@users.noreply.github.com"`

### 方式 B：手动 git 命令

```bash
cd website
git init -b main
git add .
git commit -m "Course site: syllabus + Lecture 1 courseware"
git remote add origin https://github.com/elysium4eva/bigdata.git
git push -u origin main
```

### 方式 C：网页上传

在 `bigdata` 仓库页面点击 *Add file → Upload files*，把本目录（含 `assets/`、`files/`）
整体拖入并提交；再到 Settings → Pages 开启 main 分支。

## 更新内容

课程内容与大纲**同源**：网站页面由课程大纲 JSON 自动生成，改一处即可同步。

```powershell
# 1) 编辑大纲内容（文字、排课、章节要点都在这里）
notepad ..\syllabus-src\syllabus-content.json
# 2) 重新生成 index.html / syllabus.html / lecture-01.html
node ..\website-src\build-site.js
# 3) 检查链接与结构
node ..\website-src\check-links.js
# 4) 提交
git add . ; git commit -m "Update content" ; git push
```

新增某章课件时：把 PDF 放进 `files/`，然后在 `website-src/build-site.js` 的
`FILES` 映射中为该讲添加条目（`{ label, href }`），再重新生成即可。

## 关于大文件

`files/` 中两个较大的 PDF 只是**下载用**，不影响页面加载：

- 若不希望仓库过大，可删除 `L01-introduction-converted.pdf`（34.7 MB）与
  `L06-logistic-regression-linear-classifier.pdf`（16.8 MB），并同步删去
  `lecture-01.html`、`index.html` 中对应的下载按钮（改 `build-site.js` 的 `FILES` 后重新生成）。
- GitHub 单文件上限 100 MB；如需管理更大的课件，可启用 **Git LFS**。

## 站点自检

```powershell
node ..\website-src\check-links.js
```

会报告本地链接是否全部存在、章节卡片与课表行数是否正确（正常应为
12 周课表、15 张章节卡片、0 个问题）。
