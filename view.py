import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re

# ================== 配置 ==================
CSV_FILE = "美国热度前100事件_10天聚类.csv"

# 扩展国家/地区词典（覆盖你的数据）
COUNTRIES = {
    '中国', '美国', '菲律宾', '加拿大', '英国', '日本', '澳大利亚', '越南', '台湾',
    '韩国', '俄罗斯', '印度', '朝鲜', '马来西亚', '印尼', '新加坡', '法国', '德国',
    '东盟', '北约', '欧盟', '联合国', '荷兰', '新西兰', '泰国', '巴西'
}
LOCATIONS = {
    '南海', '台海', '黄岩岛', '仁爱礁', '美济岛', '永暑礁', '巴士海峡',
    '太平洋', '东海', '冲绳', '关岛', '夏威夷', '东南亚'
}
ORGANIZATIONS = {
    '国防部', '五角大楼', '白宫', '国务院', '外交部', '中国海警', '解放军',
    'CNN', 'BBC', '新华社', '路透社', '彭博社', '纽约时报', '华盛顿邮报',
    '东盟', '北约', '联合国', '金砖国家', 'NRDC','美国国防部长','英国皇家空军','澳大利亚皇家海军','菲律宾驻华盛顿特使',
    '美国卫生与公众服务部','中国共产党','美国总统','菲律宾海岸警卫队','中国船只','加拿大皇家海军','美国海军'
}
# PEOPLE = {
#     '拜登', '特朗普', '布林肯', '奥斯汀', '沙利文', '王毅', '秦刚', '小马科斯',
#     '岸田文雄', '阿尔巴内塞', '普京', '泽连斯基', '拉米', 'Stefanie Spear'
# }

EVENT_TYPE_KEYWORDS = {
    '联合巡航': ['巡航', '航行自由', '军舰', '航母', '驱逐舰', '舰队','演习','军演','军事演习','合作','海试'],
    '补给/建设': ['补给', '建设', '填海', '基建', '驻守', '物资', '驳船', '登陆演习'],
    '执法对峙': ['海警', '执法', '对峙', '拦截', '驱离', '登船', '冲突','对抗','攻击','军事行动'],
    '外交声明': ['声明', '抗议', '谴责', '表态', '外交', '照会', '言论', '交锋','威胁','警告'],
    '舆论视频': ['视频', '曝光', '直播', '社交媒体', 'Twitter', 'X.com', '照片']
}

# ================== 数据加载与清洗 ==================

def extract_urls_from_row(row):
    """从第7列开始的所有链接列中提取去重URL"""
    urls = set()
    for col in row.index[7:]:
        cell = str(row[col]).strip()
        if not cell or cell == 'nan':
            continue
        # 处理 "url1";"url2" 格式
        parts = cell.replace('"', '').split(';')
        for part in parts:
            part = part.strip()
            if part.startswith('http'):
                urls.add(part)
    return list(urls)

def classify_event_type(text):
    text = str(text).lower()
    for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
        if any(kw.lower() in text for kw in keywords):
            return event_type
    return '其他'

@st.cache_data
def load_and_process_data():
    # 读取 CSV（自动跳过空列）
    df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
    
    # 重命名列（确保前7列正确）
    base_cols = ['主事件标题', '事件标题', '时间', '地点', '涉事方', '关键动作', '总触达量']
    if len(df.columns) >= 7:
        df = df.iloc[:, :107]  # 截断到合理长度（含链接）
        df.columns = base_cols + [f'链接{i}' for i in range(1, len(df.columns)-6)]
    else:
        st.error("CSV 列数不足")
        st.stop()

    # 解析时间 & 触达量
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['总触达量'] = pd.to_numeric(df['总触达量'], errors='coerce').fillna(0).astype(int)
    df = df.dropna(subset=['时间']).sort_values('时间').reset_index(drop=True)

    # 提取所有证据链接
    df['证据链接列表'] = df.apply(extract_urls_from_row, axis=1)

    # 添加事件类型
    df['事件类型'] = df['事件标题'].apply(classify_event_type)

    return df

# ================== 实体提取 ==================

def extract_entities_from_row(row):
    text = f"{row['事件标题']} {row['关键动作']} {row['涉事方']} {row['地点']}"
    entities = {
        '国家/地区': [e for e in COUNTRIES if e in text],
        '地点': [e for e in LOCATIONS if e in text],
        '组织': [e for e in ORGANIZATIONS if e in text],
    }
    return entities

# ================== 主应用 ==================

def main():
    st.set_page_config(page_title="地缘政治事件聚类分析系统", layout="wide")
    st.title("🌍 地缘政治事件聚类分析系统")

    try:
        df = load_and_process_data()
    except Exception as e:
        st.error(f"❌ 加载数据失败：{e}")
        st.stop()

    # ===== 侧边栏筛选 =====
    st.sidebar.header("🔍 全局筛选")
    min_date = df['时间'].min().date()
    max_date = df['时间'].max().date()
    start_date, end_date = st.sidebar.date_input(
        "📅 时间范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    all_locations = sorted(set(loc.strip() for locs in df['地点'].str.split('、') for loc in locs if loc.strip()))
    selected_loc = st.sidebar.multiselect("📍 地点", all_locations)

    all_parties = sorted(set(p.strip() for parties in df['涉事方'].str.split('、') for p in parties if p.strip()))
    selected_party = st.sidebar.multiselect("👥 涉事方", all_parties)

    # 过滤
    filtered_df = df[
        (df['时间'].dt.date >= start_date) &
        (df['时间'].dt.date <= end_date)
    ]
    if selected_loc:
        filtered_df = filtered_df[filtered_df['地点'].str.contains('|'.join(selected_loc), na=False)]
    if selected_party:
        filtered_df = filtered_df[filtered_df['涉事方'].str.contains('|'.join(selected_party), na=False)]

    if filtered_df.empty:
        st.warning("⚠️ 当前筛选条件下无数据")
        st.stop()

    total_reach = filtered_df['总触达量'].sum()
    st.sidebar.metric("📊 总触达量", f"{total_reach:,}")
    # ===== 功能 1：年度节点表（按10天窗口聚合触达量）=====
    st.header("1️⃣ 年度节点表")
    window_size_days = 10
    filtered_df['window_start'] = filtered_df['时间'].dt.floor(f'{window_size_days}D')
    node_table = filtered_df.groupby('window_start').agg(
        热度=('总触达量', 'sum'),
        事件数=('事件标题', 'count'),
        Top事件=('主事件标题', lambda x: '；'.join(sorted(set(x))[:3]))
    ).reset_index()
    node_table['窗口结束'] = node_table['window_start'] + pd.Timedelta(days=window_size_days - 1)

    # ✅ 关键修改：将时间列格式化为 YYYY-MM-DD 字符串（不带小时）
    node_table['window_start'] = node_table['window_start'].dt.strftime('%Y-%m-%d')
    node_table['窗口结束'] = node_table['窗口结束'].dt.strftime('%Y-%m-%d')

    node_table = node_table.sort_values('window_start', ascending=False)
    st.dataframe(node_table[['window_start', '窗口结束', '热度', '事件数', 'Top事件']], use_container_width=True)
    
   
    # ===== 功能 1.5：事件类型时间趋势（累积）=====
    st.header("📈 事件类型时间趋势（10天窗口 · 累积）")
    trend_df = filtered_df.copy()
    trend_df['window'] = pd.to_datetime(trend_df['时间']).dt.to_period('10D').dt.start_time
    type_trend = trend_df.groupby(['window', '事件类型']).size().reset_index(name='事件数')
    type_trend = type_trend.sort_values('window')

    # 计算每个事件类型的累积和
    type_trend['累计事件数'] = type_trend.groupby('事件类型')['事件数'].cumsum()

    fig_trend = px.line(
        type_trend,
        x='window',
        y='累计事件数',
        color='事件类型',
        title="各类事件数量随时间变化（10天窗口 · 累积）",
        markers=True
    )
    fig_trend.update_layout(
        xaxis_title="时间",
        yaxis_title="累计事件数量",
        hovermode="x unified",
        yaxis=dict(tickformat=',d')  # 强制 Y 轴为整数（不带小数）
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ===== 功能 2.5：主事件影响力分布（按时间）=====
    st.header("📊 主事件影响力分布（按首次出现时间）")
    main_event_summary = filtered_df.groupby('主事件标题').agg(
        子事件数=('事件标题', 'count'),
        总触达量=('总触达量', 'sum'),
        首次出现=('时间', 'min')
    ).reset_index()

    # 合并事件类型（取最常见的）
    type_mode = filtered_df.groupby('主事件标题')['事件类型'].agg(
        lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else '其他'
    ).reset_index()
    main_event_summary = main_event_summary.merge(type_mode, on='主事件标题')

    # 确保“首次出现”是 datetime 类型（便于 Plotly 处理）
    main_event_summary['首次出现'] = pd.to_datetime(main_event_summary['首次出现'])

    fig_bubble = px.scatter(
        main_event_summary,
        x='首次出现',               # ← 改为日期
        y='总触达量',
        size='总触达量',
        color='事件类型',
        hover_name='主事件标题',
        hover_data={
            '首次出现': '|%Y-%m-%d',  # ← 关键：悬停只显示年月日
            '子事件数': True,
            '总触达量': ':,'
        },
        title="主事件影响力分布（X轴 = 首次出现日期）",
        size_max=60
    )
    fig_bubble.update_layout(
        xaxis_title="首次出现日期",
        yaxis_title="总触达量",
        xaxis=dict(tickformat='%Y-%m-%d')  # 可选：X轴刻度也显示为日期
    )
    st.plotly_chart(fig_bubble, use_container_width=True)


    # ===== 功能 2：事件卡片库（两级结构：主事件 → 子事件）=====
    st.header("2️⃣ 事件卡片库（按主事件聚合）")
    
    # 按主事件分组
    grouped = filtered_df.groupby('主事件标题', sort=False)
    
    for main_event, group in grouped:
        # 计算该主事件的总触达量和子事件数
        total_sub_reach = group['总触达量'].sum()
        sub_count = len(group)
        
        # 按时间排序子事件
        group = group.sort_values('时间')
        # 对当前主事件的子事件按时间排序（升序：最早在前）
        group_sorted = group.sort_values('时间')
        first_date = group_sorted['时间'].iloc[0].strftime('%Y-%m-%d')
        total_sub_reach = group['总触达量'].sum()
        sub_count = len(group)

        with st.expander(f"🗓️ {first_date} | 📁 {main_event} | 🔥 总触达量: {total_sub_reach:,} | 📌 {sub_count} 条子事件"):
            # 如果子事件较多，可考虑加个提示
            if sub_count > 5:
                st.caption(f"共 {sub_count} 条子事件，按时间倒序展示")
            
            # 倒序展示（最新在上）
            for _, row in group[::-1].iterrows():
                st.markdown(f"#### 🗓️ {row['时间'].strftime('%Y-%m-%d')} | 🔥 {row['总触达量']:,}")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**标题**：{row['事件标题']}")
                    st.markdown(f"**摘要**：{row['关键动作']}")
                    st.markdown(f"**地点**：{row['地点']}")
                    st.markdown(f"**涉事方**：{row['涉事方']}")
                    # 显示证据链接
                    urls = row['证据链接列表']
                    if urls:
                        st.markdown("**🔗 证据链接**：")
                        for url in urls[:5]:  # 最多显示5个
                            st.markdown(f"- [{url}]({url})")
                        if len(urls) > 5:
                            st.caption(f"... 还有 {len(urls)-5} 个链接")
                with col2:
                    ents = extract_entities_from_row(row)
                    st.markdown("**涉及实体**")
                    for cat, items in ents.items():
                        if items:
                            st.markdown(f"- **{cat}**：{', '.join(set(items))}")
                st.divider()  # 分隔线

    # ===== 功能 3：实体榜 =====
    st.header("3️⃣ 高频实体榜")
    entity_counter = Counter()
    for _, row in filtered_df.iterrows():
        ents = extract_entities_from_row(row)
        for cat, items in ents.items():
            entity_counter.update([(cat, item) for item in items])

    if entity_counter:
        top_entities = entity_counter.most_common(20)
        ent_df = pd.DataFrame(top_entities, columns=['(类别, 实体)', '频次'])
        ent_df[['类别', '实体']] = pd.DataFrame(ent_df['(类别, 实体)'].tolist(), index=ent_df.index)
        ent_df = ent_df[['类别', '实体', '频次']]

        for category in ['国家/地区', '地点', '组织']:  # 注意：你已注释掉 PEOPLE，所以去掉 '人物'
            cat_data = ent_df[ent_df['类别'] == category].head(6)
            if not cat_data.empty:
                fig = px.bar(
                    cat_data, y='实体', x='频次', orientation='h',
                    title=f"🔥 高频{category}",
                    height=300
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("未识别到预定义实体")

    # ===== 功能 4：可介入窗口建议 =====
    st.header("4️⃣ 可介入窗口建议（按事件类型）")
    type_summary = filtered_df.groupby('事件类型').agg(
        事件数=('事件标题', 'count'),
        总触达量=('总触达量', 'sum')
    ).reset_index().sort_values('总触达量', ascending=False)
    st.dataframe(type_summary, use_container_width=True)

    st.markdown("""
    **类型说明**：
    - **联合巡航**：军舰行动、航行自由
    - **补给/建设**：岛礁建设、驳船、登陆演习
    - **执法对峙**：海警驱离、海上冲突
    - **外交声明**：官方表态、言语交锋
    - **舆论视频**：社交媒体视频/图片曝光
    """)

if __name__ == "__main__":

    main()
