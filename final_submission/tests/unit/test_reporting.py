from uavx.autonomy.reporting import ReportingEngine


def test_report_is_on_time():
    r = ReportingEngine(10.0)
    r.begin("t0", 1, 100.0)
    out = r.transmit("t0", 105.0, True, 25.0)

    assert out.report_success
    assert out.status == "ON_TIME"
    assert out.pdr == 1.0


def test_report_is_late():
    r = ReportingEngine(10.0)
    r.begin("t0", 1, 100.0)
    out = r.transmit("t0", 111.0, True, 25.0)

    assert out.status == "LATE"


def test_packet_loss():
    r = ReportingEngine(10.0)
    r.begin("t0", 1, 100.0)
    out = r.transmit("t0", 105.0, False, 25.0)

    assert not out.report_success
    assert out.pdr == 0.0
