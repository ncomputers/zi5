<?php
$r = new Redis();
$r->connect('127.0.0.1', 6379);
$start = isset($_GET['start']) ? $_GET['start'] : '';
$end = isset($_GET['end']) ? $_GET['end'] : '';
$times = $ins = $outs = $current = [];
if ($start && $end) {
    $start_ts = strtotime($start);
    $end_ts = strtotime($end);
    $entries = $r->lrange('history', 0, -1);
    foreach ($entries as $item) {
        $e = json_decode($item, true);
        $ts = isset($e['ts']) ? intval($e['ts']) : 0;
        if ($ts < $start_ts || $ts > $end_ts) continue;
        $times[] = date('Y-m-d H:i', $ts);
        $in = isset($e['in']) ? intval($e['in']) : 0;
        $out = isset($e['out']) ? intval($e['out']) : 0;
        $ins[] = $in;
        $outs[] = $out;
        $current[] = $in - $out;
    }
}
?>
<!DOCTYPE html>
<html>
<head>
    <title>Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="p-4">
<div class="container" style="max-width:900px">
    <h1 class="mb-3">Report</h1>
    <form class="row g-2 mb-4">
        <div class="col-auto"><input type="datetime-local" name="start" class="form-control" value="<?php echo htmlspecialchars($start); ?>" required></div>
        <div class="col-auto"><input type="datetime-local" name="end" class="form-control" value="<?php echo htmlspecialchars($end); ?>" required></div>
        <div class="col-auto"><button class="btn btn-primary" type="submit">Load</button></div>
    </form>
    <canvas id="chart" height="300"></canvas>
</div>
<script>
const data={
    labels: <?php echo json_encode($times); ?>,
    datasets:[
        {label:'In',yAxisID:'y1',data:<?php echo json_encode($ins); ?>,borderColor:'green',tension:0.2},
        {label:'Out',yAxisID:'y1',data:<?php echo json_encode($outs); ?>,borderColor:'red',tension:0.2},
        {label:'Currently Inside',yAxisID:'y2',data:<?php echo json_encode($current); ?>,borderColor:'blue',tension:0.2}
    ]
};
new Chart(document.getElementById('chart'),{
    type:'line',
    data:data,
    options:{scales:{y1:{type:'linear',position:'left'},y2:{type:'linear',position:'right',grid:{drawOnChartArea:false}}}}
});
</script>
</body>
</html>
