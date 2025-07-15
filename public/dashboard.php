<?php
$r = new Redis();
$r->connect('127.0.0.1', 6379);
$inKeys = $r->keys('cam:*:in');
$outKeys = $r->keys('cam:*:out');
$in = 0;
foreach ($inKeys as $k) { $in += intval($r->get($k)); }
$out = 0;
foreach ($outKeys as $k) { $out += intval($r->get($k)); }
$current = $in - $out;
$cfg = json_decode($r->get('config'), true);
$max = isset($cfg['max_capacity']) ? intval($cfg['max_capacity']) : 0;
?>
<!DOCTYPE html>
<html>
<head>
    <title>Crowd Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-4">
<div class="container text-center">
    <h1 class="mb-4">Crowd Dashboard</h1>
    <div class="row justify-content-center">
        <div class="col-md-3">
            <div class="alert alert-success">
                <h4>Currently Inside</h4>
                <div class="display-6"><?php echo $current; ?></div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="alert alert-primary">
                <h4>Total Entered</h4>
                <div class="display-6"><?php echo $in; ?></div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="alert alert-danger">
                <h4>Total Exited</h4>
                <div class="display-6"><?php echo $out; ?></div>
            </div>
        </div>
    </div>
</div>
</body>
</html>
