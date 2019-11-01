$( document ).ready(function() {
    
    var boardName = 'myBoard'
    //  Hard code start configuration
	var positions = {}
	positions['e4'] = 'bK'
	positions['f4'] = 'wP'
    
    //  Initialize board configuration
	var boardConfig = {
		pieceTheme: '/static/{piece}.png',
		position:positions,
		draggable: true,
		dropOffBoard: 'trash',
		onSnapEnd: runWhenPieceMoves,
        onDragStart: clearBackground,
		sparePieces: true
	}
		
    
    //  SVG for text below chess board
	var svgW = 400
	var svgH = 100
	var svgTextSize = 15
	var svg = d3.select('#boardConfig')
		.append('svg')
		.attr("width", svgW)
		.attr("height", svgH)		
		
	var positionText = svg.append('text')
		.attr('x',0)
		.attr('y', 0)
		.attr('dominant-baseline', 'hanging')
		.attr('text-anchor', 'start')  
		.style('font-size', svgTextSize + 'px')
    
    var valueText = svg.append('text')
		.attr('x',0)
		.attr('y', svgTextSize)
		.attr('dominant-baseline', 'hanging')
		.attr('text-anchor', 'start')  
		.style('font-size', svgTextSize + 'px')
    
    //  Create chess board
	var board = Chessboard(boardName,boardConfig)
    
    var colorBarWidth = 400
    var colorBarHeight = 20
    var color = d3.interpolateViridis
    var colorBar = d3.select('#' + boardName)
        .append('svg')
        .attr('width', colorBarWidth)
        .attr('height',colorBarHeight)
    
    var numBars = 100
    for (var b = 0; b < numBars; b++) {
        colorBar.append('rect')
        .attr('x', b * colorBarWidth/numBars)
        .attr('y', 0)
        .attr('width', colorBarWidth/numBars)
        .attr('height',20)
        .style('fill',color(b/numBars))
        
    }
    
            
    function clearBackground(){
        var boardPositions = board.position()
        var squares = Object.keys(boardPositions)
        var num = squares.length
        for (var s = 0; s < num; s++) {
                test = d3.select('#' + boardName).select('.square-' + squares[s])
                    .style('background-color','')
            }
        
    }

    //  Function ran when piece moves
	function runWhenPieceMoves(){		
		var boardPositions = board.position()
		positionText.text("Positions:\t" + JSON.stringify(boardPositions).replace(/\"/g,'').replace(/\{/g,'').replace(/\}/g,'').replace(/\,/g,', '))
		$.post( "/postmethod", boardPositions, function(err, req, resp){
            var pieceValues = resp.responseJSON
            valueText.text("Values:\t" + JSON.stringify(pieceValues).replace(/\"/g,'').replace(/\{/g,'').replace(/\}/g,'').replace(/\,/g,', '))
            var squares = Object.keys(pieceValues)
            var values = Object.values(pieceValues)
            var num = squares.length
            d3.select('#' + boardName).selectAll('.square').style('background-color','')
            for (var s = 0; s < num; s++) {
                test = d3.select('#' + boardName).select('.square-' + squares[s])
                    .style('background-color',color(values[s]))
            }
            
    });
	}
	
	// Run Function
    runWhenPieceMoves()
  
});
