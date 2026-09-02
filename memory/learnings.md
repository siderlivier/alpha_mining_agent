# 因子挖掘經驗庫
（agent於Distill回合維護，證據編號附各條，置信等級[single_case|corroborated|multi_case|promoted]，[失效]不刪除。本輪：cheap+quality併入規則7、legs可交易性併入規則1，補margin_d21/法人流量/股利政策證據，審計併入規則9，移除已證死佇列項，刪逐筆日誌與DSL明細，新增streak型別修正、clip_std提案。）

## 全域規則
1. [promoted] 多頭腿可交易性優先於IC/ICIR：判死前先查多頭腿年化超額>0，IC/ICIR顯著不代表leg可交易，最常見隱性拒絕原因。sdiv比值型複合尤易legs不單調(A-0911)，優先mul(rank,rank)取代sdiv。證據26+。
2. [promoted] 產業限定敘事系統性預測失準：industry_icir常與敘事相反(A-0843/A-0898)；金融/生技反轉敘事連續失敗。證據35+。
3. [promoted] if_else(cond,X,0)零值填補在cs_rank下同分並列壓低IC/ICIR；優先連續加權或提案rank_nz。證據10+。
4. [promoted] 短窗(21日)delta/if_else訊號換手率常>0.4致stage4拒絕；應疊加ts_mean(2-3)平滑或改63日以上窗口。證據16+(A-0903)。
5. [promoted] 產業獲利趨勢類(ROE/margin ts_slope)因子高機率與F-001/F-002/F-011重複，新案先查相關性(A-0908)。
6. [promoted] 週轉率/借券/券資比冷卻類因子系統性與F-008高相關(ρ0.5-0.9)，提交前必查。證據15+(A-0864)。
7. [promoted] cs_rank(A)×cs_rank(B)複合公式是最大死因池，系統性與F-002/F-008/F-011等既有家族相關0.5-0.85；換單一輸入不足以逃脫，需改骨幹結構。「估值×品質」子型態本輪6/15死於此(A-0923)；industry_demean/換窗期表層變形同樣無效(A-0916)。證據30+(A-0839,A-0895,A-0898)。
8. [門檻] Stage1約ICIR 0.45；Stage2約|ρ|>0.5；邊界(0.5-0.6)排入殘差化重測，不判死。
9. [audit] AST深度3衰減20% vs 深度2衰減-3%；操作子衰減sign 43%>delta 34%>sub 31%>mul 13%>cs_rank 12%>ts_std 3%；技術面(42%)遠高於基本面(9%)，優先淺層AST、避免sign/delta/sub作頂層運算子。證據n=25。
10. [promoted] 符號反轉優先於判死：方向穩定但與假設相反時，優先套neg()重提而非判假設全滅。驗證3次(A-0868,A-0906,A-0907)。
11. [multi_case] margin_d21核心複合因子系統性偏弱(不論交乘對象/平滑窗口ICIR持續<0.1)，先驗證原始欄位獨立IC否則不建議續推。證據：A-0865,A-0879,A-0891,A-0896,A-0899,A-0903。
12. rank相減系統性弱化訊號，集中多頭腿alpha假設應避免相減式(A-0901)；分母(淨利/自身波動度)在高波動股不穩，改用資產營收基準，sdiv無下限保護放大雜訊(A-0902)。
13. [multi_case] 法人流量弱訊號：frgn/trust_net_21原始或無基本面錨點交乘持續偏弱、方向不穩，優先權下修。證據10+(A-0369,A-0787,A-0878,A-0880,A-0883,A-0888等)。
14. [multi_case] 股利政策訊號弱：殖利率類指標連續4次(A-0866,A-0871,A-0901,A-0911)ICIR<0.2或legs不單調，除非徹底改變機制角度不建議短期內再投入。
15. [方法論] direction欄位描述最終formula(含neg轉換後)方向而非原始直覺(A-0877)；decay_pct於ICIR近零時被異常放大，優先看mean_ic與逐年一致性(A-0905)。
- [red_ocean_boundary|multi_case] 營收/毛利改善的組合式因子（delta/rev_yoy/gross_margin交叉）極易與庫內F-009、F-011高相關(ρ>0.55)；同賽道再提案須先做正交化或換全新確認維度。證據：C-2, C-3。（attempt: A-0927）
- [long_short_asymmetry|multi_case] 本輪多個候選（C-2、C-6、C-7、C-8、C-15）出現短腳超額>長腳超額（甚至長腳為負），顯示訊號實質是「避開什麼」而非「該買什麼」；台股放空受限，長腳超額≤0時即使rank IC/ICIR過關也不可交易，篩選階段應提早檢查long_excess_ann方向。（attempt: A-0931）
- [表達模式|multi_case] 用sign(x)對另一因子做連續乘法加權(而非if_else清零)在兩個獨立案例中都把IC壓到接近零，優先懷疑是表達方式壓縮了資訊而非機制失效。證據：C-3, C-4。（attempt: A-0942）
- [語法深度限制|multi_case] 深度上限3常在「時序轉換(delta/ts_mean/ts_std)再包兩層cs_rank相乘」的四層結構卡住(too_deep)，假設完全未被測試就出局；產生器應優先把雙特徵時序轉換拆成先各自算好單一transform再rank相乘，避免同時疊加combo運算子。證據：C-2、C-7、C-13。（attempt: A-0954）
- [基本面兩兩相乘紅海|multi_case] 「兩個基本面欄位(獲利率/成長率/ROE等)各自cs_rank後相乘」這類簡單組合，本輪4次獨立嘗試(C-1 op_margin×frgn_ratio、C-3 gross_margin_delta×rev_yoy、C-6 industry_demean_mom×rev_yoy、C-11 net_margin_delta×roe_delta)全部因與庫內因子ρ0.59~0.80撞期被拒；僅將level換成delta只能小幅降相關性(約-0.07)，不足以脫離紅海，需换掉共用核心欄位或改變組合邏輯（如三因子交乘、產業內排序）才可能區隔。證據：C-1、C-3、C-6、C-11。（attempt: A-0964）
- [紅海邊界|multi_case] 低流動性/低週轉率作為訊號核心區別因子，容易與F-008(低調接近新高)高相關(ρ>0.5)，即使表面敘事不同(借券分歧、超跌反轉)。證據：C-3(ρ0.64)、C-15(ρ0.55)。（attempt: A-0971）
- [紅海邊界|multi_case] 「營收成長×品質過濾(毛利/accruals/margin)」因子空間紅海中心是F-011，非F-006/F-012/F-022，提交前應優先檢查與F-011相關性。證據：C-4(ρ0.60)、C-9(ρ0.52)。（attempt: A-0972）
- [產業歸因檢核|multi_case] 宣稱「產業限定」機制的假說，提交前應核對industry_icir分布是否真在該產業最強；本輪C-10(宣稱金融但電子最強、缺金融數據)、C-11(宣稱生技但生技最弱、金融最強)皆錯配。（attempt: A-0978）
- [紅海|multi_case] 價值(bp/ep)×獲利品質(roe/margin)的交乘因子，無論用水準、趨勢(delta/ts_slope)或加信用交易確認腿，都與 F-002/R-016/R-022 高相關(ρ 0.60–0.81)；金融限定變體另還打不到自身 industry_icir 門檻。證據：C-10, C-12, C-15。（attempt: A-0998）
- [紅海邊界-成長持續性|multi_case] streak_true(基本面成長/獲利為正的月數) 類因子 standalone 常表現佳,但經濟內涵=長期穩定高品質股,實測與庫內 價值×成長(F-003, ρ0.75)、ROE(R-016, ρ0.66) 高相關而死於 stage2;新設此類因子前先預檢價值×成長/品質族或先殘差化再評估。證據：C-1, C-14。（attempt: A-0999）
- [構造反模式-成長率相減|multi_case] sub(a_yoy, b_yoy)(如營運槓桿差、本業純度)在分母近零時數值爆量,sub-train 訊號微弱、validation 劇烈衰減或反向;營運槓桿/獲利品質改用『利潤率變化量』『Δ對 Δ 的滾動迴歸斜率』或 level 型指標。證據：C-6, C-13。（attempt: A-1004）
- [因子構造|multi_case] streak_true(X > ts_med(X,24)) 型『持續站上自身中位數』因子與 delta(X,12) 高度共線(上升序列恆高於落後中位數)，且樣本內 ICIR 嚴重灌水(C-5 in-sample 0.73→val -0.01)；提交前必對對應年變化因子殘差化並檢視 validation。證據：C-5, C-12, C-13。（attempt: A-1018）
- [品質因子|multi_case] 台股靜態品質『水準』因子(毛利率、現金轉換率，未做變化或交互)呈防禦型：跌月 IC > 漲月 IC、多頭腿≈0或為負，非可交易多頭 alpha。品質類應走變化量、產業內相對、或與價格創高/低關注交互。證據：C-8, C-9。（attempt: A-1021）

## 禁忌方向
- 周轉率×營收交乘方向反覆或近零；摩擦延續(軋空/借券/回補)與超跌反彈均值回歸皆系統性失敗；short_margin_ratio負向版仍duplicate失敗(A-0887)，方向窮盡。
- 外資持股水準/趨勢認證系統性偏弱；法人買超/賣超無基本面錨點方向不穩或近零(含63日平滑x基本面確認A-0870/A-0880)。
- 金融業「不透明→訊號更強」失敗(含bp斜率天真資本累積假設A-0920)；生技「燒錢是好事」反轉、事件後波動回落敘事(A-0922)連續失敗。
- 個股估值時序回歸2013-14後轉負；季頻財務差分型失效；純動能(mom_240)偏弱；避地雷類因子多頭腿常≤0；delta(X,12)對yoy欄位再取delta雙重差分近乎零。
- mom_20減mom_120台股一致負IC；Amihud式abs(mom_N)/amt反向IC(已測真實定義A-0863)勿再試。
- [corroborated] neg(mom_240)為核心疊加基本面確認仍呈反轉特徵、多頭腿超額為負，不可交易。證據：A-0919,A-0885。
- ts_slope疊加已是動能性質欄位(如rev_mom)雙重平滑滯後(A-0912)，避免「動能的動能」直接疊加ts_slope。
- [dividend_signal_dead|multi_case] 股利/殖利率相關因子（水準、穩定度、變動delta）累計5次獨立嘗試ICIR均未過門檻或驗證期反轉，判定此類別在現有資料下機制性失效，非必要不再嘗試同族群變體。證據：C-15（含提案自述之前4次失敗）。（attempt: A-0938）
- [紅海邊界|single_case] 「低關注度代理×獲利率」(C-1 vs F-020, ρ0.67)與「股價貼近52週高點」(C-6 vs F-016, ρ0.67)兩個因子家族在庫內飽和度高，僅調整平滑/排名方式難以達成|ρ|<0.5，需更換核心變數才可能通過stage2。證據：C-1, C-6。（attempt: A-0944）
- [禁忌|multi_case] 以低獲利率(毛利率/營益率/淨利率)作為『市場過度看空、將向上重定價』的正向訊號代理，一律無效或反向；margin 水準本身為正向品質因子，不可押其反轉。證據：C-3, C-8, C-9。（attempt: A-0992）
- [中期價格動能-多頭|multi_case] 台股 2012-2019,中期價格動能(mom_60/mom_120)單獨、加基本面過濾或加路徑平順度 overlay,sub-train IC 皆 ≤0 且下跌月更差,屬反轉而非動能;此方向多頭腿暫勿再投入。證據：C-2, C-3, C-9。（attempt: A-1000）
- [品質因子|single_case] 台股『低 EPS 年增率波動=品質溢酬』方向反轉，每年每產業 IC 為負(ICIR −0.51)。不要再以低盈餘波動當多頭 alpha 提交。證據：C-14。（attempt: A-1027）

## 紅海地圖
- 低波動(neg(vol_21/63))易與F-005/F-018重疊，含industry_demean變形版(A-0916)；創高動能+獲利成長確認易與F-016/F-017重疊。
- rev_yoy相關組合已飽和(F-011/F-013/F-018，含ts_rank(rev_yoy,24)單欄位版A-0900)；accruals已被F-006佔滿。
- 低關注度代理(成交金額/外資比/股息/週轉率低)×獲利/價值已被F-020高度飽和，換代理仍ρ>0.6(A-0863,A-0909)；需換非交易活躍度代理(法人持股/分析師覆蓋)才可能突圍。
- 「相對自身滾動歷史高點」轉換已與F-008/F-023重疊。cash_backed_value_orphan全市場版與電子限定版皆已被F-019覆蓋(A-0895,A-0910)，已窮盡。
- overlooked_profitability_liquidity(低成交金額×本業獲利率，電子限定)已通過三階段驗證——全庫罕見passed案例。opex_investment_intensity負向穩健但alpha集中空頭腿，宜作濾網而非獨立因子。

## 待驗證假設佇列（優先級由高到低）
2. [Q-012] 對F-001/F-002/F-008/F-011/F-012/F-022家族殘差化後重測複合因子。
3. [Q-013] div_yield+獲利穩定度：mul(cs_rank(div_yield),cs_rank(neg(ts_std(eps_yoy,12))))取代sdiv重驗legs；已連續4次失敗(規則14)，優先度下修。
5. [Q-015] sector_rotation_relative_value：修正direction標籤後重提(前次敗於expression_bad非假設本身)。
6. [Q-016] 金融業結構性欄位(逾放比/呆帳覆蓋率/利差)測金融限定敘事；dps/股利年增量欄位測股利政策。
7. [Q-017] 借券/期貨稀疏覆蓋率欄位限定「覆蓋率>0.7子集」內排序重測。
8. [Q-018] 生技毛利躍升：移除eps_yoy成分(改rev_yoy或毛利率絕對門檻)，降低與F-022殘餘重疊(前次0.66仍偏高)。
9. [Q-019] streak(cond,n)/clip_std(x,n)/industry_demean(x)落地後，重測3個型別問題陣亡的連續確認假說及產業限定假說。
10. [Q-020] [流程] Stage2邊界相關度(0.5-0.6)一律先殘差化重測，不直接判死。
- [Q-021] C-4：把 rank_nz(if_else(greater(rev_yoy,0), op_income_yoy, 0)) 改為分組內排名（僅在rev_yoy>0子集中做cs_rank後再與全池對齊）或用 sign(rev_yoy) 加權而非清零，避免子集被壓成同值拖累coverage。
- [Q-022] C-11：把 mul(cs_rank(ts_slope(gross_margin,12)), cs_rank(ts_rsq(gross_margin,12))) 中的rsq窗口拉長到24個月或改用ts_rsq的等級門檻（如僅排除最差20%而非連續乘法加權），降低對訓練期特定平穩走勢的過度依賴。
- [Q-023] C-13：把 ts_rank(px_hi252,24) 改為 ts_mean(px_hi252,24) 或直接用px_hi252搭配較短的ts_mean窗口，保留貼近高點的幅度資訊，避免二次排名稀釋訊號。
- [Q-024] 「營收驗證獲利/毛利改善」類假設(C-3, C-4)若要保留，建議把sign(rev_yoy)乘法權重換成if_else條件邏輯(rev_yoy>0時取值、否則設為極端劣評分而非乘以-1)，或改用rev_yoy本身的cs_rank做交乘而非sign二值化，重新測試是否恢復訊號。
- [Q-025] 「毛利趨勢真確度」(C-5)機制或許仍成立，建議改用if_else(ts_rsq(gross_margin,24)>閾值, cs_rank(ts_slope(gross_margin,12)), 中性值)的門檻式表達取代mul(cs_rank,cs_rank)交乘，降低過擬合風險後重測validation穩定性。
- [Q-026] 「金融應計連續性」(C-10)建議把formula限定在金融產業子樣本內計算(如乘以is_financial遮罩或僅對金融股做cs_rank)，排除電子/半導體等產業的反向雜訊後重新驗證sub-train ICIR。
- [Q-031] 護城河穩健股：簡化為 mul(cs_rank(gross_margin), cs_rank(F-007))（直接引用既有的neg(ts_std(rev_mom,6))欄位取代重新巢狀）重新提交，同時需重新檢查與F-004/F-007的相關性。
- [Q-003] 【解鎖】當股價短期已轉強(mom_20>0)時，高券資比代表大量空頭部位承受虧損壓力，若股價續強將觸發強制回補形成自我強化的軋空上漲；若股價仍偏弱，高券資比只是單純看空未必反轉，因此需要動能轉強作為觸發條件才具軋空預測力，這與『（原受限於 DSL，rank_nz 已實作；原公式 cs_rank(if_else(greater(mom_20,0), short_margin_ratio, 0))，證據 A-0281）
- [Q-004] 【解鎖】中期動能(mom_120)訊號品質參差不齊，可能來自基本面改善也可能來自炒作雜訊；當動能出現同時伴隨外資近月買超(frgn_net_21>0)，代表此動能較可能有基本面研究支撐（外資交易通常伴隨產業研究），用法人流向篩選（原受限於 DSL，rank_nz 已實作；原公式 cs_rank(if_else(greater(frgn_net_21,0), mom_120, 0))，證據 A-0353）
- [Q-007] 【解鎖】多數生技公司仍處燒錢研發階段，投資人對整個板塊套用高風險/難獲利刻板印象；即使少數生技公司已達成真實且盈餘品質乾淨的獲利，市場仍給予類股風險折價而非重新歸類，此類股標籤黏著性造成系統性低估。（原受限於 DSL，industry_demean 已實作；原公式 mul(cs_rank(ep), cs_rank(neg(accruals)))，證據 A-0652）
- [Q-010] 【解鎖】現金殖利率相對於EPS年增率波動度的比值，代表配息背後獲利的穩定程度；多數投資人只看殖利率高低、不校正獲利穩定性，使「風險調整後殖利率」高的公司被低估，金融股受法規要求穩定配息文化下此機制應更顯著。（原受限於 DSL，clip_std 已實作；原公式 cs_rank(sdiv(div_yield, ts_std(eps_yoy,12)))，證據 A-0823）
- [Q-038] 落後補漲:把 less(mom_60,0) 硬 gate 改為連續負動能權重,或對 mom_60 殘差化 eps_yoy,將覆蓋率從 0.37 拉高後重測 PEAD 未反應。
- [Q-039] 毛利價背離:改用 cs_rank(delta(gross_margin,12)) 與 cs_rank(neg(mom_120)) 的排名交乘,或對 mom_120 殘差化毛利改善量,取代 if_else 硬切,避免退化為 R-018 並解決覆蓋率 0.34 的問題。
- [Q-040] 融資緩降浮額沉澱：改測『margin_d21 的 12 月斜率為負 且 單月變動截尾標準差低』的交乘，並限縮中小型股樣本

## DSL 表達力受限訊號
- if_else零值填補製造同分並列(規則3)→已提案rank_nz。缺產業內排序/遮罩運算子(A-0469,A-0615,A-0652,A-0845)→已提案industry_demean(x)。
- 缺日曆條件運算子，無法限定財報更新月(A-0421,A-0745,A-0326)。
- streak型別衝突：簽名要求數值而非布林condition，3個連續確認假說皆卡在greater()輸出型別不符(A-0913,A-0915,A-0918)→本輪重提案streak(cond,n)。
- 缺winsorize/clip截尾，sdiv/yoy低基期分母易爆量(A-0522,A-0823,A-0902)→本輪提案clip_std(x,n)。
- ts_corr/ts_slope對季頻forward-fill欄位無法區分獨立觀測數(A-0300,A-0700)；缺真AND邏輯，mul(rank,rank)只能連續加權稀釋(A-0433,A-0845,A-0776)。
- 其餘：非單調關係(A-0517)、產業分類僅4類(A-0469)、巢狀深度上限3(A-0501)、ts_slope疊加變化率欄位雙重平滑無法評估訊噪比(A-0912)。

## 近期 attempts 聚合（n=150，A-0774~A-0923；已消化為上方規則，不再列明細）
verdict：stage1 57(38%)、stage2 57(38%)、stage4 32(21%)、syntax 3(2%)、passed 1(1%)
failure_type(n=135)：duplicate 52(39%)、hypothesis_wrong 45(33%)、expression_bad 37(27%)、none 1
高頻category(≥3次)：cash_backed_value_orphan 5、overlooked_profitability_liquidity 4、domestic_institutional_flow_confirmation 3、revenue_momentum_continuation 3