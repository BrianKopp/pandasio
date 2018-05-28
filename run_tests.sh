#!/bin/bash

source venv/pandasio/bin/activate

echo "Executing tests..."
echo ""
coverage erase

coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_binary
coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_datetime_utils
coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_numpy_compression
coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_numpy_decompression
coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_numpy_float_compression
coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_numpy_float_rounding
coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_numpy_utils
coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_pandas_utils
coverage run -a --omit "venv/*" -m pandasio.utils.tests.test_validation

coverage run -a --omit "venv/*" -m pandasio.tests.test_pandabar
coverage run -a --omit "venv/*" -m pandasio.tests.test_pandabar_details_bytes

report_coverage=false
include_missing=false
for i in "$@"
do
case $i in
    r|-r)
    report_coverage=true
    shift
    ;;
    m|-m)
    include_missing=true
    shift
    ;;
    *)
    shift
    ;;
esac
done

if $report_coverage
then
    echo "Coverage report:"
    if $include_missing
    then
        coverage report -m
    else
        coverage report
    fi
    echo "End of coverage report."
fi

deactivate
echo "Tests completed."