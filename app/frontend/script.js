"use strict";

const byId = id => document.getElementById(id);
const yesNo = ["Yes", "No"];
const internetOptions = [...yesNo, "No internet service"];
const groups = {
  profile: [
    ["gender", "Gender", ["Male", "Female"], "Female"],
    ["SeniorCitizen", "Senior citizen", [0, 1], 0],
    ["Partner", "Partner", yesNo, "Yes"],
    ["Dependents", "Dependents", yesNo, "No"]
  ],
  services: [
    ["PhoneService", "Phone service", yesNo, "Yes"],
    ["MultipleLines", "Multiple lines", [...yesNo, "No phone service"], "No"],
    ["InternetService", "Internet service", ["DSL", "Fiber optic", "No"], "Fiber optic"],
    ["OnlineSecurity", "Online security", internetOptions, "No"],
    ["OnlineBackup", "Online backup", internetOptions, "No"],
    ["DeviceProtection", "Device protection", internetOptions, "No"],
    ["TechSupport", "Tech support", internetOptions, "No"],
    ["StreamingTV", "Streaming TV", internetOptions, "Yes"],
    ["StreamingMovies", "Streaming movies", internetOptions, "No"]
  ],
  billing: [
    ["tenure", "Tenure (months)", null, 5],
    ["Contract", "Contract", ["Month-to-month", "One year", "Two year"], "Month-to-month"],
    ["PaperlessBilling", "Paperless billing", yesNo, "Yes"],
    ["PaymentMethod", "Payment method", ["Electronic check", "Mailed check",
      "Bank transfer (automatic)", "Credit card (automatic)"], "Electronic check"],
    ["MonthlyCharges", "Monthly charges", null, 85.5],
    ["TotalCharges", "Total charges (optional)", null, 420.75]
  ]
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

for (const [group, fields] of Object.entries(groups)) {
  for (const [name, title, options, initial] of fields) {
    const label = element("label", title);
    const input = element(options ? "select" : "input");
    input.name = input.id = name;
    input.required = name !== "TotalCharges";
    if (options) {
      for (const value of options) {
        const caption = name === "SeniorCitizen" ? ["No", "Yes"][value] : value;
        input.add(new Option(caption, value));
      }
    } else {
      input.type = "number";
      input.min = 0;
      input.step = name === "tenure" ? "1" : "any";
      if (name === "tenure") input.max = 100;
    }
    input.value = initial;
    label.append(input);
    byId(group).append(label);
  }
}

function syncServices() {
  function restrict(id, noService, inactive) {
    const field = byId(id);
    for (const option of field.options) option.disabled = (option.value === noService) !== inactive;
    if (field.selectedOptions[0].disabled) field.value = inactive ? noService : "No";
  }
  restrict("MultipleLines", "No phone service", byId("PhoneService").value === "No");
  for (const [name, , options] of groups.services) {
    if (options === internetOptions) restrict(name, "No internet service", byId("InternetService").value === "No");
  }
}
syncServices();
byId("services").addEventListener("change", syncServices);

async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = body?.detail;
      const message = Array.isArray(detail)
        ? detail.map(item => item.loc.slice(1).join(".") + ": " + item.msg).join(" · ")
        : detail;
      throw new Error(message || "API request failed (HTTP " + response.status + ").");
    }
    if (!body) throw new Error("The API returned an empty or invalid response.");
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Request timed out. Please try again.");
    if (error instanceof TypeError) throw new Error("Could not reach FastAPI. Check the connection and try again.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function refreshHealth() {
  let healthy = false;
  try { healthy = (await request("/health")).model_loaded === true; } catch { /* Show unavailable below. */ }
  byId("health").classList.toggle("ok", healthy);
  byId("health").querySelector("span").textContent = healthy ? "Model online" : "API unavailable";
}
refreshHealth();
setInterval(refreshHealth, 60000);

const percent = value => (value * 100).toFixed(1) + "%";
const count = value => value.toLocaleString("en-US");
const result = byId("result");
const emptyResult = () => {
  result.className = "";
  result.replaceChildren(element("h2", "Prediction result"),
    element("p", "Run a prediction to see the churn probability and decision threshold."));
};
let version = 0;
byId("customer").addEventListener("input", () => { version++; emptyResult(); });

byId("customer").addEventListener("submit", async event => {
  event.preventDefault();
  const button = byId("submit");
  if (button.disabled) return;
  const submitted = version;
  const payload = Object.fromEntries(new FormData(event.currentTarget));
  for (const key of ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]) {
    payload[key] = payload[key] === "" ? null : Number(payload[key]);
  }
  button.disabled = true;
  button.textContent = "Predicting…";
  result.className = "";
  result.replaceChildren(element("h2", "Prediction result"), element("p", "Waiting for the model…"));
  try {
    const data = await request("/predict", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)});
    if (submitted !== version) return;
    const {churn_prediction: decision, churn_probability: probability, threshold} = data;
    if (!["Yes","No"].includes(decision) || ![probability, threshold].every(n => Number.isFinite(n) && n >= 0 && n <= 1)) {
      throw new Error("The response is missing a valid prediction, probability, or threshold.");
    }
    result.className = decision === "Yes" ? "high" : "low";
    result.replaceChildren(element("h2", decision === "Yes" ? "Churn predicted" : "Retention predicted"),
      element("div", (probability * 100).toFixed(2) + "%", "score"), element("p", "Churn probability"),
      element("hr"), element("p", "Churn prediction: " + decision + " · Decision threshold: " + threshold));
    result.focus();
  } catch (error) {
    if (submitted === version) result.replaceChildren(element("h2", "Prediction unavailable"),
      element("p", error.message, "error"));
  } finally {
    button.disabled = false;
    button.replaceChildren("Predict churn ", element("span", "→"));
  }
});

function card(title) {
  const node = element("article");
  node.append(element("h2", title));
  return node;
}

function bar(label, value, total) {
  const node = element("section", undefined, "bar");
  const caption = element("div");
  caption.append(element("span", label), element("strong", count(value) + " · " + percent(value / total)));
  const track = element("section", undefined, "track");
  const fill = element("span", undefined, "fill");
  fill.style.width = percent(value / total);
  track.append(fill);
  node.append(caption, track);
  return node;
}

function table(headers, rows, activeIndex = -1) {
  const node = element("table");
  const head = node.createTHead().insertRow();
  headers.forEach(title => head.append(element("th", title)));
  const body = node.createTBody();
  rows.forEach((values, index) => {
    const row = body.insertRow();
    if (index === activeIndex) row.className = "active";
    values.forEach(value => row.insertCell().textContent = value);
  });
  return node;
}

const series = [["precision","Precision","#215ee8"],["recall","Recall","#00847a"],["f1","F1","#8a55b6"]];
function sweepChart(rows) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 600 215");
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Precision, recall, and F1 by threshold. Exact values follow in the data table.");
  function draw(tag, attrs, text) {
    const node = document.createElementNS(svg.namespaceURI, tag);
    Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key,value));
    if (text) node.textContent = text;
    svg.append(node);
  }
  const first = rows[0].threshold, last = rows[rows.length - 1].threshold;
  const x = v => 48 + (v - first) / (last - first || 1) * 524;
  const y = v => 177 - v * 155;
  for (const value of [0,.25,.5,.75,1]) {
    draw("line", {x1:48,x2:572,y1:y(value),y2:y(value),stroke:"#dce5f0"});
    draw("text", {x:39,y:y(value)+4,"text-anchor":"end",fill:"#53677c","font-size":12}, value*100+"%");
  }
  for (const row of rows) draw("text", {x:x(row.threshold), y:201,"text-anchor":"middle",fill:"#53677c","font-size":13},row.threshold.toFixed(2));
  for (const [key,,color] of series) {
    draw("polyline", {points:rows.map(r => x(r.threshold)+","+y(r[key])).join(" "), fill:"none",stroke:color,"stroke-width":3});
    for (const r of rows) draw("circle",{cx:x(r.threshold),cy:y(r[key]),r:3.5,fill:color});
  }
  return svg;
}

function renderReport(data) {
  const perf = card("Model performance");
  const metrics = element("div", undefined, "metrics");
  for (const [key,title] of [["accuracy","Accuracy"],["precision","Precision"],["recall","Recall"],["f1","F1-score"],["roc_auc","ROC-AUC"]]) {
    const item = element("div", undefined, "metric");
    const value = data.metrics[key];
    item.append(element("small", title), element("strong", value == null ? "—" : key === "roc_auc" ? value.toFixed(3) : percent(value)));
    metrics.append(item);
  }
  perf.append(metrics, element("p","Evaluation threshold: " + data.metrics.threshold + " · Accuracy: " + data.accuracy_source));
  const dataset = card("Customer distribution");
  dataset.append(element("div",count(data.customers),"score"), element("small","Total customers"),
    bar("Retained",data.customers-data.churned,data.customers),bar("Churned",data.churned,data.customers));
  const contracts = card("Contract distribution");
  for (const [name,n] of Object.entries(data.contracts)) contracts.append(bar(name,n,data.customers));
  contracts.append(element("p","Average tenure: " + data.mean_tenure.toFixed(1) + " months · Average monthly charges: " + data.mean_monthly.toFixed(2)));
  const pair = element("div",undefined,"pair");
  pair.append(dataset,contracts);
  const sweep = card("Threshold analysis");
  sweep.append(element("p","How the classification cutoff affects the model’s precision, recall, and F1-score."));
  const legend = element("div",undefined,"legend");
  for (const [,name,color] of series) {
    const key = element("span"), dot = element("i");
    dot.style.background = color;
    key.append(dot,name);
    legend.append(key);
  }
  if (data.sweep.length) sweep.append(sweepChart(data.sweep),legend);
  const details = element("details");
  details.append(element("summary","Threshold data"),table(["Threshold","Precision","Recall","F1-score"],
    data.sweep.map(r => [r.threshold.toFixed(2),percent(r.precision),percent(r.recall),percent(r.f1)]),
    data.sweep.findIndex(r => r.threshold === data.threshold)));
  sweep.append(details);
  const model = card("Model information");
  model.append(element("p","XGBoost · Decision threshold " + data.threshold + " · Training time " + data.training.train_time.toFixed(2) + " s"));
  if (!data.best_params) model.append(element("p","Tuning results unavailable. The table shows the parameters used for training."));
  model.append(table(["Training parameter","Value"],Object.entries(data.training.params)));
  if (data.best_params) model.append(table(["Best parameter","Tuned value"],Object.entries(data.best_params)));
  byId("report").replaceChildren(perf,pair,sweep,model);
}

async function loadReport() {
  const report = byId("report");
  report.replaceChildren(element("p","Loading analytics…"));
  try { renderReport(await request("/analytics")); }
  catch (error) {
    const retry = element("button","Retry analytics");
    retry.onclick = loadReport;
    report.replaceChildren(element("p",error.message,"error"),retry);
  }
}

function navigate() {
  const analytics = location.hash === "#analytics";
  byId("predict").hidden = analytics;
  byId("analytics").hidden = !analytics;
  byId("crumb").textContent = analytics ? "Analytics" : "Prediction";
  document.title = byId("crumb").textContent + " · Telco Churn";
  for (const link of document.querySelectorAll("nav a")) {
    if (link.hash === (analytics ? "#analytics" : "#predict")) link.setAttribute("aria-current","page");
    else link.removeAttribute("aria-current");
  }
  if (analytics) loadReport();
}
window.addEventListener("hashchange", navigate);
navigate();
