$( document ).ready(function() {
    
	//	Board name that matches div
    var boardName = 'myBoard'
	
    //  Hard code start configuration
	var positions = {}
	positions['e4'] = 'bK'
	positions['f4'] = 'wP'
    
    //  Initialize board configuration
	var boardConfig = {
		pieceTheme: '/static/{piece}.png',	//	location of piece images
		position:positions,					//	initial piece positions dictionary
		draggable: true,					//	pieces are draggable
		dropOffBoard: 'trash',				//	What happens when piece is dragged off board
		onSnapEnd: onSnapEnd,				//	Function to call after piece is snapped to a new position
		sparePieces: true					//	Draws spare black and white pieces on either side of the board
	}
    
    //  SVG for text below chess board
	var svgW = 400
	var svgH = 100
	var svgTextSize = 15
	var svg = d3.select('#boardConfig')
		.append('svg')
		.attr("width", svgW)
		.attr("height", svgH)		
	
	//	Text object for listing positions
	var positionText = svg.append('text')
		.attr('x',0)
		.attr('y', 0)
		.attr('dominant-baseline', 'hanging')
		.attr('text-anchor', 'start')  
		.style('font-size', svgTextSize + 'px')
    
	//	Text object for listing values
    var valueText = svg.append('text')
		.attr('x',0)
		.attr('y', svgTextSize)
		.attr('dominant-baseline', 'hanging')
		.attr('text-anchor', 'start')  
		.style('font-size', svgTextSize + 'px')
    
    //  Create chess board
	var board = Chessboard(boardName,boardConfig)
    
	//	Colorbar
    var colorBarWidth = 400
    var colorBarHeight = 20
	var numBars = 100
    var color = d3.interpolateViridis
    var colorBar = d3.select('#' + boardName)
        .append('svg')
        .attr('width', colorBarWidth)
        .attr('height',colorBarHeight)
    for (var b = 0; b < numBars; b++) {
        colorBar.append('rect')
        .attr('x', b * colorBarWidth/numBars)
        .attr('y', 0)
        .attr('width', colorBarWidth/numBars)
        .attr('height',20)
        .style('fill',color(b/numBars))
        
    }
	
	//	Clears custom color under a single square
	function clearSquare(square){
		$('#myBoard .square-' + square).css('background', '')
	}

    //  Function ran when piece moves
	function onSnapEnd(source=null, target=null, piece=null){	
		//	Get current board positions
		var boardPositions = board.position()
		
		//	Update positions text object
		positionText.text("Positions:\t" + JSON.stringify(boardPositions).replace(/\"/g,'').replace(/\{/g,'').replace(/\}/g,'').replace(/\,/g,', '))
		
		//	POST to python and receive values
		$.post( "/postmethod", boardPositions, function(err, req, resp){
			
			//	Receive piece values dictionary
            var pieceValues = resp.responseJSON
			
			//	Update values text object
            valueText.text("Values:\t" + JSON.stringify(pieceValues).replace(/\"/g,'').replace(/\{/g,'').replace(/\}/g,'').replace(/\,/g,', '))
			
			//	Loop over piece values and draw colors in appropriate squares
            for (var s in pieceValues) {
				$('#myBoard .square-' + s).css('background', color(pieceValues[s]))
            }
		});
		
		//	If piece isn't added from spare pieces and source isn't null, clear source square
		//	source is only null in manual run of function below, to initialize the values
		if (source != null && source != 'spare') {
			clearSquare(source)
		}
	}
	
	// Run Function
    onSnapEnd()
  
});
