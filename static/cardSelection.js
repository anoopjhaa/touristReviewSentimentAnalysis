$('.card').click(function () {
    var target = $(this).data('target');
    console.log(target)
    $('.card-details').hide();
    $(target).show();
});