document.addEventListener("DOMContentLoaded", function() {
    
    // --- DOM 요소 가져오기 ---
    const optimizeButton = document.getElementById('optimize-button');
    
    // [수정] jQuery 객체로 Select2 요소를 가져옵니다.
    const $stockSelect = $('#stock-multiselect'); 
    
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessage = document.getElementById('error-message');
    const resultArea = document.getElementById('portfolio-result-area');
    
    // ... (나머지 결과 표시 영역 변수들은 동일) ...
    
    const exchangeRateInfo = document.getElementById('exchange-rate-info');
    const aiReason = document.getElementById('ai-reason');
    const predictionTableBody = document.getElementById('prediction-table-body');
    const sentimentScore = document.getElementById('sentiment-score');
    const sentimentTotal = document.getElementById('sentiment-total');
    const sentimentPositive = document.getElementById('sentiment-positive');
    const sentimentNegative = document.getElementById('sentiment-negative');
    const sentimentNeutral = document.getElementById('sentiment-neutral');
    const sentimentStatus = document.getElementById('sentiment-status');
    const pieChartCanvas = document.getElementById('portfolio-pie-chart');
    const markowitzHeader = document.getElementById('markowitz-header-row');
    const markowitzData = document.getElementById('markowitz-data-row');
    let portfolioPieChart = null; 

    const FLASK_BASE_URL = "http://127.0.0.1:5000"; 

    // --- 1. 페이지 로드 시 종목 목록 불러오기 ---
    async function loadStockList() {
        try {
            const response = await fetch(`${FLASK_BASE_URL}/api/get-stock-list`);
            if (!response.ok) throw new Error("종목 목록 로딩 실패");
            
            const data = await response.json();
            
            const stockSelectEl = document.getElementById('stock-multiselect');
            stockSelectEl.innerHTML = ''; // "로딩 중..." 옵션 제거
            
            data.all_stocks.forEach(stockName => {
                const option = document.createElement('option');
                option.value = stockName;
                option.textContent = stockName;
                stockSelectEl.appendChild(option);
            });

            // [신규] Select2 라이브러리를 초기화합니다.
            $stockSelect.select2({
                placeholder: "종목을 선택하세요 (클릭하여 추가)",
                allowClear: true
            });
            
            // [신규] API에서 받은 기본 추천 종목을 Select2에 설정합니다.
            $stockSelect.val(data.default_stocks).trigger('change');

        } catch (error) {
            console.error("Stock list error:", error);
            document.getElementById('stock-multiselect').innerHTML = `<option value="">${error.message}</option>`;
        }
    }
    
    // --- 2. 최적화 버튼 클릭 이벤트 ---
    optimizeButton.addEventListener('click', async () => {
        
        // 2.1. [수정] Select2에서 선택된 종목 가져오기
        const selectedOptions = $stockSelect.val(); 
        
        if (!selectedOptions || selectedOptions.length < 2) { // selectedOptions가 null일 수 있음
            showError("최소 2개 이상의 종목을 선택해야 합니다.");
            return;
        }
        
        // 2.2. UI 초기화
        showLoading(true);
        showError(null);
        resultArea.style.display = 'none';

        try {
            // 2.3. Flask API 호출
            const response = await fetch(`${FLASK_BASE_URL}/api/optimize-portfolio`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stocks: selectedOptions }) // 수정된 selectedOptions 사용
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "서버에서 오류가 발생했습니다.");
            }
            
            // 2.4. 결과 표시
            displayResults(result);
            resultArea.style.display = 'block';

        } catch (error) {
            console.error("최적화 실패:", error);
            showError(`오류 발생: ${error.message}. (Flask 서버가 실행 중인지 확인하세요)`);
        } finally {
            showLoading(false);
        }
    });

    // --- 3. 결과 표시 함수 (수정 없음) ---
    function displayResults(result) {
        // (이하 displayResults 함수 내용은 이전과 동일하게 둡니다)
        const exchangeRate = result.exchange_rate || 1400.0;
        exchangeRateInfo.textContent = `적용 환율: 1 USD = ${exchangeRate.toLocaleString('ko-KR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} KRW`;

        predictionTableBody.innerHTML = ''; 
        const predictions = result.model_prediction.individual || {};
        for (const [name, pred] of Object.entries(predictions)) {
            if (!pred) continue; 
            const direction = pred.direction === 1 ? '<span class="text-success">상승 ▲</span>' : '<span class="text-danger">하락 ▼</span>';
            const currentPriceKrw = formatPrice(pred.current_price, pred.currency, exchangeRate);
            
            const row = `<tr>
                <td>${name}</td>
                <td>${currentPriceKrw}</td>
                <td>${direction}</td>
            </tr>`;
            predictionTableBody.innerHTML += row;
        }

        const sentiment = result.sentiment_analysis || {};
        const summary = sentiment.summary || {};
        sentimentScore.textContent = (sentiment.sentiment_score || 0).toFixed(3);
        sentimentTotal.textContent = `${summary.total || 0}개`;
        sentimentPositive.textContent = `${summary.positive || 0}개`;
        sentimentNegative.textContent = `${summary.negative || 0}개`;
        sentimentNeutral.textContent = `${summary.neutral || 0}개`;
        sentimentStatus.textContent = `AI 종합 판단: ${sentiment.status || 'N/A'}`;

        const markowitz = result.markowitz_portfolio || {};
        const m_weights = markowitz.weights || {};
        
        markowitzHeader.innerHTML = ''; // 기존 헤더 초기화
        markowitzData.innerHTML = '';   // 기존 데이터 초기화

        if (Object.keys(m_weights).length > 0) {
            for (const [name, weight] of Object.entries(m_weights)) {
                markowitzHeader.innerHTML += `<th>${name}</th>`;
                markowitzData.innerHTML += `<td>${(weight * 100).toFixed(2)}%</td>`;
            }
        } else {
            markowitzHeader.innerHTML = '<th>데이터 없음</th>';
        }
        
        const finalPortfolio = result.final_portfolio || {};
        const weights = finalPortfolio.final_weights || {};
        
        const filteredLabels = Object.keys(weights).filter(key => weights[key] > 0);
        const filteredData = filteredLabels.map(key => weights[key]);

        if (portfolioPieChart) {
        portfolioPieChart.destroy();
        portfolioPieChart = null; 
    }
        
        if (filteredLabels.length > 0) {
            portfolioPieChart = new Chart(pieChartCanvas, {
                type: 'pie',
                data: {
                    labels: filteredLabels,
                    datasets: [{
                        data: filteredData,
                        backgroundColor: [ 
                            '#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', 
                            '#6f42c1', '#fd7e14', '#6610f2'
                        ],
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'top' },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.label || '';
                                    let value = context.raw || 0;
                                    return `${label}: ${(value * 100).toFixed(2)}%`;
                                }
                            }
                        }
                    }
                }
            });
        } else {
        const context = pieChartCanvas.getContext('2d');
        context.clearRect(0, 0, pieChartCanvas.width, pieChartCanvas.height);
    }
        aiReason.textContent = finalPortfolio.reason || "AI 동적 비중 조정이 비활성화되었습니다.";
    }

    // --- 4. 헬퍼 함수 (수정 없음) ---
    function showLoading(isLoading) {
        // (이하 동일)
        loadingSpinner.style.display = isLoading ? 'block' : 'none';
        optimizeButton.disabled = isLoading;
    }

    function showError(message) {
        // (이하 동일)
        if (message) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
        } else {
            errorMessage.style.display = 'none';
        }
    }
    
    function formatPrice(price, currency, exchangeRate) {
        // (이하 동일)
        if (currency === 'KRW') {
            return `${Math.round(price).toLocaleString('ko-KR')}원`;
        } else {
            const convertedPrice = price * exchangeRate;
            return `${Math.round(convertedPrice).toLocaleString('ko-KR')}원 (${price.toFixed(2)} ${currency})`;
        }
    }

    // --- 5. 초기 함수 실행 ---
    loadStockList();

});