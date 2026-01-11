function loadDashboardCharts() {
    const pie = document.getElementById("diseasePie");
    if (!pie) return;

    new Chart(pie, {
        type: "pie",
        data: {
            labels: ["No Heart Disease", "Heart Disease"],
            datasets: [{
                data: [60, 40],
                backgroundColor: ["#2ecc71", "#e74c3c"]
            }]
        }
    });
}

function loadResultChart(risk) {
    const chart = document.getElementById("riskChart");
    if (!chart) return;

    new Chart(chart, {
        type: "doughnut",
        data: {
            labels: ["Risk", "Safe"],
            datasets: [{
                data: [risk, 100 - risk],
                backgroundColor: ["#e63946", "#2ecc71"]
            }]
        }
    });
}
